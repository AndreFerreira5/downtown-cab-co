import mlflow
import os
from dotenv import load_dotenv

load_dotenv()


def configure_mlflow():
    MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:3030")
    mlflow.set_tracking_uri(MLFLOW_URI)
