import logging
from .train import run_training
from .test import test_models
from .config import TrainingConfig
from .logging_config import configure_logging

# configure logging globally
configure_logging()
logger = logging.getLogger(__name__)


def train(training_config: TrainingConfig):
    info = run_training(
        commit_sha=training_config.COMMIT_SHA,
        model_name=training_config.MODEL_NAME,
        experiment_name=training_config.EXP_NAME,
    )
    return {"message": "train finished", **info}


if __name__ == "__main__":
    train(TrainingConfig())
    test_models()
