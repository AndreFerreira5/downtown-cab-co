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
import src.inference_api.data.processer as inference_processer
import numpy as np
from .logging_config import configure_logging
from .config import InferenceConfig
import types
import sys

# configure logging globally
configure_logging()
logger = logging.getLogger(__name__)


class TrendResidualModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["regressor"], "rb") as f:
            self.trend_model = pickle.load(f)
        with open(context.artifacts["booster"], "rb") as f:
            self.booster_model = pickle.load(f)

    def predict(self, context, model_input):
        model_input["tpep_pickup_datetime"] = pd.to_datetime(model_input["tpep_pickup_datetime"])
        X, _ = preprocess_taxi_data(
            pd.DataFrame([model_input]),
            remove_outliers=False,
            create_features=True
        )

        trend_pred = self.trend_model.predict(X[['date_int', 'sin_time', 'cos_time']])
        trend_pred = np.maximum(trend_pred, 1.0)

        X.drop(columns=['trip_duration', 'tpep_pickup_datetime', 'date_int', 'sin_time', 'cos_time'], errors='ignore')
        X_residual = X.select_dtypes(include=['number', 'category'])
        ratio_pred = self.booster_model.predict(X_residual)

        final_pred = trend_pred * ratio_pred
        return final_pred

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


class PredictRequest(BaseModel):
    data: List[Dict[str, Any]]


def load_model_into_app():
    logger.info(f"Connecting to MLflow at {inference_config.MLFLOW_URI}...")
    try:
        app.state.model = mlflow.pyfunc.load_model(
            #model_uri=f"models:/{inference_config.MODEL_NAME}@{inference_config.MODEL_ALIAS}"
            model_uri=f"models:/{inference_config.MODEL_NAME}@staging" # TODO this is just for prediction testing, remove it later and use the appropriate  aliases
        )
        logger.info("Model loaded successfully.")
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
        load_model_into_app()
        raise HTTPException(
            status_code=503, detail="Model not loaded. Call /reload or /train first."
        )

    try:
        return {"predictions": [app.state.model.predict(model_input=trip) for trip in req.data]}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction error")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
