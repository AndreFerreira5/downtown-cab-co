import logging
from .train import run_hyperparameter_tuning, run_training
from .config import TrainingConfig
from .logging_config import configure_logging
import mlflow

# configure logging globally
configure_logging()
logger = logging.getLogger(__name__)


def train(training_config: TrainingConfig):
    mlflow.set_experiment(training_config.EXP_NAME)

    # first get the best hyperparameters
    model_params = run_hyperparameter_tuning(training_config.COMMIT_SHA, training_config.MODEL_NAME)

    # only then train the model with the best hyperparameters
    return run_training(model_params, training_config.COMMIT_SHA, training_config.MODEL_NAME)


if __name__ == "__main__":
    regressor, booster = train(TrainingConfig())
