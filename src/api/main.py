import logging
import os

import uvicorn
from fastapi import FastAPI, HTTPException
import mlflow
import mlflow.pyfunc

from typing import List, Optional
from pydantic import BaseModel
import pandas as pd
from .train import run_training

from .logging_config import configure_logging

# TODO this is while traininng algorithm isnt implemented
try:
    from .train import run_training  # type: ignore
    TRAIN_AVAILABLE = True
except Exception:
    TRAIN_AVAILABLE = False

# configure logging globally
configure_logging()
logger = logging.getLogger(__name__)

# ----- Config -----
MODEL_NAME = os.getenv("MODEL_NAME", "nyc_taxi_duration")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5050")  # its mlflow and not localhost because the
                                                                     # container will be named mlflos
os.environ["MLFLOW_ALLOWED_HOSTS"] = "*"  # allow container->container calls (per tutorial)

mlflow.set_tracking_uri(MLFLOW_URI)

app = FastAPI(title="NYC Taxi Baseline API", version="0.1.0")

# Keep the model in memory (reloaded on /reload or after /train)
app.state.model = None

def load_model_into_app():
    try:
        app.state.model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
        return True
    except Exception as e:
        print(f"[WARN] Could not load model: {e}")
        app.state.model = None
        return False


class PredictRequest(BaseModel):
    columns: List[str]
    data: List[List[float]]


class TrainRequest(BaseModel):
    # Path to a CSV/Parquet with (at least) pickup_datetime, dropoff_datetime, trip_distance
    data_path: str
    experiment_name: Optional[str] = "nyc_taxi_baseline"


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": app.state.model is not None, "mlflow_uri": MLFLOW_URI}


@app.post("/train")
def train(req: TrainRequest):

    if not TRAIN_AVAILABLE:
        raise HTTPException(status_code=501, detail="Training endpoint is disabled until an algorithm is chosen.")

    """
    Trains a simple baseline model and logs everything to MLflow.
    Registers best run as a model and sets alias to 'production'.
    """
    info = run_training(
        data_path=req.data_path,
        experiment_name=req.experiment_name,
        model_name=MODEL_NAME,
        alias=MODEL_ALIAS,
    )
    # try to load the fresh model
    load_model_into_app()
    return {"message": "training_done", **info}


@app.get("/reload")
def reload_model():
    ok = load_model_into_app()
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to load model from MLflow.")
    return {"message": "model_reloaded", "alias": MODEL_ALIAS}


@app.post("/predict")
def predict(req: PredictRequest):
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Call /reload or /train first.")
    df = pd.DataFrame(req.data, columns=req.columns)
    preds = app.state.model.predict(df)
    # Ensure vanilla types for JSON
    return {"predictions": [float(x) for x in preds]}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
