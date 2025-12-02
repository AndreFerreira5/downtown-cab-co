import mlflow
import os
from dotenv import load_dotenv

load_dotenv()


class TrainingConfig:
    def __init__(self):
        self.COMMIT_SHA = os.getenv("COMMIT_SHA")
        if not self.COMMIT_SHA:
            raise EnvironmentError("Missing required env var: COMMIT_SHA")

        self.MODEL_NAME = os.getenv('MLFLOW_MODEL_NAME')
        if not self.MODEL_NAME:
            raise EnvironmentError("Missing required env var: MLFLOW_MODEL_NAME")

        self.EXP_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME')
        if not self.EXP_NAME:
            raise EnvironmentError("Missing required env var: MLFLOW_EXPERIMENT_NAME")

        self.MLFLOW_URI = os.getenv('MLFLOW_TRACKING_URI')
        if not self.EXP_NAME:
            raise EnvironmentError("Missing required env var: MLFLOW_TRACKING_URI")
        mlflow.set_tracking_uri(self.MLFLOW_URI)
