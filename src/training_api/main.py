import logging
import os

import uvicorn
from fastapi import FastAPI, HTTPException
import mlflow
import mlflow.pyfunc

from typing import List
from pydantic import BaseModel
from .train import run_training
from .config import configure_mlflow

from .logging_config import configure_logging

# configure logging globally
configure_logging()
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "nyc_taxi_duration")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")
configure_mlflow()
MLFLOW_URI = mlflow.get_tracking_uri()

app = FastAPI(title="NYC Taxi Baseline API", version="0.1.0")

# Keep the model in memory (reloaded on /reload or after /train)
app.state.model = None


def load_model_into_app():
    try:
        app.state.model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        )
        return True
    except Exception as e:
        print(f"[WARN] Could not load model: {e}")
        app.state.model = None
        return False


class PredictRequest(BaseModel):
    columns: List[str]
    data: List[List[float]]


class TrainRequest(BaseModel):
    commit_sha: str
    model_name: str
    experiment_name: str


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": app.state.model is not None,
        "mlflow_uri": MLFLOW_URI,
    }


@app.post("/train")
def train(req: TrainRequest):
    """
    Trains a simple baseline model and logs everything to MLflow.
    Registers best run as a model and sets alias to 'production'.
    """
    info = run_training(
        commit_sha=req.commit_sha,
        model_name=MODEL_NAME,
        experiment_name=req.experiment_name,
    )
    # try to load the fresh model
    load_model_into_app()
    return {"message": "train finished", **info}


@app.get("/reload")
def reload_model():
    ok = load_model_into_app()
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to load model from MLflow.")
    return {"message": "model_reloaded", "alias": MODEL_ALIAS}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
