import logging
import os

import uvicorn
from fastapi import FastAPI, HTTPException
import mlflow
import mlflow.pyfunc
import subprocess

from typing import List, Optional
from pydantic import BaseModel
import pandas as pd
from .train import run_training
from .config import configure_mlflow

from .logging_config import configure_logging

# TODO this is while traininng algorithm isnt implemented
try:
    from .train import run_training  # type: ignore

    TRAIN_AVAILABLE = True
except:
    TRAIN_AVAILABLE = False

# configure logging globally
configure_logging()
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "nyc_taxi_duration")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5050")
configure_mlflow()

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
    # data_path: str
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
        # data_path=req.data_path,
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


@app.post("/download-dataset")
def download_dataset():
    script_path = "data/one_time_data_pull.sh"
    try:
        # start subprocess with bash
        process = subprocess.Popen(
            ['bash', script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # combine stderr with stdout
            text=True,
            bufsize=1,  # line-buffered
            universal_newlines=True
        )

        # stream output line by line
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                yield output.rstrip() + '\n'  # yield line with newline

        # check return code
        retval = process.poll()
        if retval != 0:
            yield f"Script exited with error code: {retval}\n"
    except FileNotFoundError:
        yield f"Error: Script '{script_path}' not found.\n"
    except Exception as e:
        yield f"Error running script: {str(e)}\n"


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
