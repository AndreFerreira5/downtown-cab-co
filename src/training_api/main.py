import logging
import pickle

from src.training_api.test import test_predictor
from .train import run_hyperparameter_tuning, run_training
from .config import TrainingConfig
from .logging_config import configure_logging
import mlflow

# configure logging globally
configure_logging()
logger = logging.getLogger(__name__)


def train(training_config: TrainingConfig):
    mlflow.set_experiment(f"{training_config.EXP_NAME}_{training_config.COMMIT_SHA}")

    # First get the   best hyperparameters
    model_params = run_hyperparameter_tuning(training_config.COMMIT_SHA, training_config.MODEL_NAME)
    # Only then train the model with the best hyperparameters
    return run_training(model_params, training_config.COMMIT_SHA, training_config.MODEL_NAME)


if __name__ == "__main__":
    regressor, booster = train(TrainingConfig())
