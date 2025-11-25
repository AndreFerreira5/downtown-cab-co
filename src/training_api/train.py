import mlflow
from river import tree, metrics as river_metrics
from src.training_api.data.loader import DataLoader
from src.training_api.data.processer import preprocess_taxi_data
import pickle
import logging
import pandas as pd
import os
import gc
from collections import namedtuple

logger = logging.getLogger(__name__)


def train_hatr(model_params):
    if not model_params:
        model_params = {"grace_period": 50, "model_selector_decay": 0.3}
    data_loader = DataLoader("training/", download_dataset=True)

    mlflow.start_run()
    mlflow.log_param("grace_period", model_params["grace_period"])
    mlflow.log_param("model_selector_decay", model_params["model_selector_decay"])

    model = tree.HoeffdingAdaptiveTreeRegressor(**model_params)
    mae = river_metrics.MAE()
    rmse = river_metrics.RMSE()

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    batch_count = 1
    while (batch := data_loader.load_next_batch()) is not None and not batch.empty:
        processed_batch, _ = preprocess_taxi_data(batch, create_features=True)
        for row in processed_batch.itertuples(index=False):
            y = getattr(row, "trip_duration", None)
            if y is None or pd.isna(y) or y <= 0:
                continue

            x = row._asdict()
            del x["trip_duration"]

            y_pred = model.predict_one(x)
            mae.update(y, y_pred)
            rmse.update(y, y_pred)
            model.learn_one(x, y)

        # log each 10 batches to avoid overhead
        if batch_count % 5 == 0:
            mlflow.log_metrics(
                {
                    "mae": mae.get(),
                    "rmse": rmse.get(),
                    "samples_processed": data_loader.batch_size * batch_count,
                },
                step=data_loader.batch_size * batch_count,
            )

        batch_count += 1
        del processed_batch
        gc.collect()

    final_model_path = os.path.join(
        checkpoint_dir,
        f"hatr_{model_params['grace_period']}_{model_params['model_selector_decay']}_final.pkl",
    )
    with open(final_model_path, "wb") as f:
        pickle.dump(model, f)
    mlflow.log_artifact(final_model_path)
    final_metrics = {
        "final_mae": mae.get(),
        "final_rmse": rmse.get(),
    }
    run_id = mlflow.active_run().info.run_id
    mlflow.end_run()
    # TODO log some plots to mlflow
    return model, final_metrics, run_id


def run_training(commit_sha: str, model_name: str, experiment_name: str):
    mlflow.set_experiment(experiment_name)
    model_params = [
        {"grace_period": gp, "model_selector_decay": msd}
        for gp in [2000, 200, 50]
        for msd in [0.99, 0.8, 0.3]
    ]

    best_model = None
    best_metrics = None
    best_run_id = None
    best_params = None

    metric_to_track = "final_rmse"
    best_metric_value = float("inf")

    # train all model combinations
    for mp in model_params:
        logger.info(f"Training with params: {mp}")
        model, final_metrics, run_id = train_hatr(mp)

        # compare and track best model
        current_metric = final_metrics[metric_to_track]

        if current_metric < best_metric_value:
            logger.info(f"New best model! {metric_to_track}: {current_metric:.4f}")
            best_metric_value = current_metric
            best_model = model
            best_metrics = final_metrics
            best_run_id = run_id
            best_params = mp
        else:
            logger.info(
                f"Model not better. Best {metric_to_track}: {best_metric_value:.4f}, Current: {current_metric:.4f}"
            )

    # register best model to MLflow Model Registry
    if best_model is not None:
        logger.info(
            f"\nRegistering best model with {metric_to_track}: {best_metric_value:.4f}"
        )

        # save best model
        best_model_path = "checkpoints/hatr_best_model.pkl"
        with open(best_model_path, "wb") as f:
            pickle.dump(best_model, f)

        # log to the best run
        with mlflow.start_run(run_id=best_run_id):
            mlflow.log_artifact(best_model_path)
            mlflow.log_metric("is_best_model", 1)

            # register model to Model Registry
            model_uri = f"runs:/{best_run_id}/hatr_best_model.pkl"

            try:
                registered_model = mlflow.register_model(
                    model_uri=model_uri, name=model_name
                )
                logger.info(
                    f"Model registered successfully ({registered_model.version})"
                )

                from mlflow.tracking import MlflowClient

                client = MlflowClient()
                # client.set_registered_model_alias(
                #    name=model_name,
                #    alias="staging",
                #    version=registered_model.version,
                # )
                client.set_registered_model_alias(
                    name=model_name,
                    alias=commit_sha,
                    version=registered_model.version,
                )

            except Exception as e:
                logger.info(f"Model already exists in registry, updating version: {e}")
                # if model name already exists, it will create a new version
                mlflow.register_model(model_uri=model_uri, name=model_name)

        logger.info("\n" + "=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Best hyperparameters: {best_params}")
        logger.info("Best metrics:")
        for metric_name, value in best_metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
        logger.info(f"Best run ID: {best_run_id}")
        logger.info("=" * 60)

        return best_model, best_metrics, best_params

    else:
        logger.info("No models were trained successfully")
        return None, None, None
