import mlflow
import os
from dotenv import load_dotenv

load_dotenv()


class InferenceConfig:
    def __init__(self):
        self.MODEL_NAME = os.getenv('MLFLOW_MODEL_NAME')
        if not self.MODEL_NAME:
            raise EnvironmentError("Missing required env var: MLFLOW_MODEL_NAME")

        self.MODEL_ALIAS = os.getenv('MLFLOW_MODEL_ALIAS')
        if not self.MODEL_ALIAS:
            raise EnvironmentError("Missing required env var: MLFLOW_MODEL_ALIAS")

        self.MLFLOW_URI = os.getenv('MLFLOW_TRACKING_URI')
        if not self.EXP_NAME:
            raise EnvironmentError("Missing required env var: MLFLOW_TRACKING_URI")
        mlflow.set_tracking_uri(self.MLFLOW_URI)
        os.environ["MLFLOW_ALLOWED_HOSTS"] = "*"
