import logging
import uvicorn
from fastapi import FastAPI, HTTPException
import mlflow
import mlflow.pyfunc
from typing import List, Dict, Any
from typing import List
from pydantic import BaseModel
import pandas as pd
import pickle
from .data.processer import preprocess_taxi_data
from .data.downloader import load_validation_data_for_date
import src.inference_api.data.processer as inference_processer
import numpy as np
from .logging_config import configure_logging
from .config import InferenceConfig
import types
import sys
import time
import os
import requests
from collections import deque

# configure logging  globally
configure_logging()
logger = logging.getLogger(__name__)

RMSE_WINDOW_SIZE = 100  # Number of predictions to track TODO CHANGE
RMSE_THRESHOLD = 10.0   # RMSE threshold to trigger retraining TODO CHANGE

class TrendResidualModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["regressor"], "rb") as f:
            self.trend_model = pickle.load(f)
        with open(context.artifacts["booster"], "rb") as f:
            self.booster_model = pickle.load(f)

    def predict(self, context, model_input):
        input_len = len(model_input)
        model_input["tpep_pickup_datetime"] = pd.to_datetime(model_input["tpep_pickup_datetime"])
        X, _ = preprocess_taxi_data(
            pd.DataFrame(model_input),
            remove_outliers=False,
            create_features=True,
            skip_validation=True
        )

        if len(X) != input_len: logger.warning(f"Input length {input_len} differs from processed data length {len(X)}!!")

        trend_pred = self.trend_model.predict(X[['date_int', 'sin_time', 'cos_time']])
        trend_pred = np.maximum(trend_pred, 1.0)

        X.drop(columns=['trip_duration', 'tpep_pickup_datetime', 'date_int', 'sin_time', 'cos_time', 'PULocationID', 'DOLocationID'], inplace=True, errors='ignore')
        X_residual = X.select_dtypes(include=['number', 'category'])
        ratio_pred = self.booster_model.predict(X_residual)

        final_pred = trend_pred * ratio_pred

        if len(final_pred) != input_len: logger.warning(f"Input length {input_len} differs from predicted data length {len(X)}!!")

        return final_pred/60

# QUICK HACK to make the pickle model work in the inference API
# TODO later remove this code and develop a code architecture that allows sharing code between training and inference properly
training_api = types.ModuleType("src.training_api")
training_api.data = types.ModuleType("src.training_api.data")
training_api.train = types.ModuleType("src.training_api.train")
training_api.train.TrendResidualModel = TrendResidualModel
training_api.data.processer = inference_processer

sys.modules["src.training_api"] = training_api
sys.modules["src.training_api.data"] = training_api.data
sys.modules["src.training_api.data.processer"] = inference_processer
sys.modules["src.training_api.train"] = training_api.train

inference_config = InferenceConfig()

app = FastAPI(title="NYC Taxi Baseline API", version="0.1.0")

# Keep the model in memory (reloaded on /reload or after /train)
app.state.model = None

# Sliding window
app.state.prediction_errors = deque(maxlen=RMSE_WINDOW_SIZE)
app.state.retraining_triggered = False

class PredictRequest(BaseModel):
    data: List[Dict[str, Any]]


def trigger_retraining_workflow():
    """Trigger GitHub Actions workflow for model retraining"""
    github_token = os.getenv("GITHUB_TOKEN")
    github_repo = os.getenv("GITHUB_REPOSITORY")

    if not github_token or not github_repo:
        logger.error("GitHub credentials not configured. Cannot trigger retraining.")
        return False

    url = f"https://api.github.com/repos/{github_repo}/actions/workflows/2-continuous-delivery.yml/dispatches"

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {github_token}"
    }

    payload = {
        "ref": "main",
        "inputs": {
            "trigger_reason": "rmse_threshold_exceeded"
        }
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 204:
            logger.info("Retraining workflow triggered successfully!")
            return True
        else:
            logger.error(f"Failed to trigger retraining: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error triggering retraining workflow: {e}")
        return False


def check_and_trigger_retraining():
    """Check if RMSE threshold is exceeded and trigger retraining"""
    if len(app.state.prediction_errors) < RMSE_WINDOW_SIZE:
        logger.info(f"Sliding window not full yet: {len(app.state.prediction_errors)}/{RMSE_WINDOW_SIZE}")
        return

    # Calculate average RMSE over the sliding window
    avg_rmse = np.sqrt(np.mean(app.state.prediction_errors))
    logger.info(f"📊 Current sliding window average RMSE: {avg_rmse:.4f}")

    if avg_rmse > RMSE_THRESHOLD and not app.state.retraining_triggered:
        logger.warning(f"⚠️ RMSE threshold exceeded! {avg_rmse:.4f} > {RMSE_THRESHOLD}")
        logger.warning("🔄 Triggering model retraining...")

        if trigger_retraining_workflow():
            app.state.retraining_triggered = True
            logger.info("Retraining flag set. Will reset after model reload.")
        else:
            logger.error("Failed to trigger retraining workflow.")

def load_model_into_app():
    logger.info(f"Connecting to MLflow at {inference_config.MLFLOW_URI}...")
    try:
        app.state.model = mlflow.pyfunc.load_model(
            #model_uri=f"models:/{inference_config.MODEL_NAME}@{inference_config.MODEL_ALIAS}"
            model_uri=f"models:/{inference_config.MODEL_NAME}@staging" # TODO this is just for prediction testing, remove it later and use the appropriate  aliases
        )
        logger.info("Model loaded successfully.")
        app.state.retraining_triggered = False
        return True
    except Exception as e:
        logger.error(f"FATAL: Could not load model: {e}")
        return False


@app.on_event("startup")
def startup_event():
    # try to load on startup so we fail fast if config is wrong
    load_model_into_app()


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": app.state.model is not None
    }


@app.get("/reload")
def reload_model():
    ok = load_model_into_app()
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to load model from MLflow.")
    return {"message": "model_reloaded"}


@app.post("/predict")
def predict(req: PredictRequest):
    if app.state.model is None:
        if not load_model_into_app():
            raise HTTPException(
                status_code=503, detail="Model not loaded. Call /reload or /train first."
            )

    try:
        start = time.time()
        input_df = pd.DataFrame(req.data)
        end_prep = time.time()
        time_prep = end_prep - start

        start = time.time()
        results = app.state.model.predict(input_df)
        end_pred = time.time()
        time_pred = end_pred - start

        logger.info(f"Prediction results: {results}")
        logger.info(f"Data prep: {time_prep:.4f} seconds - Prediction: {time_pred:.4f} seconds")
        return {"predictions": results.tolist()}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction error")


@app.post("/validate")
def validate_model():
    """Periodic validation using historical 2013 data"""
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        current_date = pd.Timestamp.now()
        validation_data = load_validation_data_for_date(current_date)

        if validation_data is None or len(validation_data) == 0:
            return {"message": "no_validation_data"}

        # Prepare data for prediction (same format as training)
        y_true = validation_data['trip_duration'].values / 60

        # Make predictions
        predictions = app.state.model.predict(validation_data)

        # Calculate squared errors
        squared_errors = (predictions - y_true) ** 2
        for se in squared_errors:
            app.state.prediction_errors.append(float(se))

        check_and_trigger_retraining()

        avg_rmse = float(np.sqrt(np.mean(app.state.prediction_errors)))
        logger.info(f"Validation: RMSE={avg_rmse:.4f}, window={len(app.state.prediction_errors)}/{RMSE_WINDOW_SIZE}")

        return {
            "validation_date": str(current_date.date()),
            "validation_month": f"2013-{current_date.month:02d}",
            "samples_validated": len(validation_data),
            "current_avg_rmse": avg_rmse,
            "window_size": len(app.state.prediction_errors),
            "threshold": RMSE_THRESHOLD,
            "retraining_triggered": app.state.retraining_triggered
        }

    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9001, reload=True)
