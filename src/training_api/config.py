import mlflow
import os


def configure_mlflow():
    MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5050")
    mlflow.set_tracking_uri(MLFLOW_URI)
