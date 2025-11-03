import mlflow
from river import tree, metrics as river_metrics
from src.training_api.data.loader import DataLoader
from src.training_api.data.processer import preprocess_taxi_data
import pickle
import logging
import pandas as pd
import os


logger = logging.getLogger(__name__)


def train_hatr(model_params):
    if not model_params:
        model_params = {'grace_period': 50, 'model_selector_decay': 0.3}
    data_loader = DataLoader("training/", download_dataset=True)

    mlflow.start_run()

    model = tree.HoeffdingAdaptiveTreeRegressor(**model_params)
    mae = river_metrics.MAE()
    rmse = river_metrics.RMSE()

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    batch_count = 1
    while (batch := data_loader.load_next_batch()) is not None and not batch.empty:
        processed_batch, _ = preprocess_taxi_data(batch, create_features=True)
        if "trip_duration" not in processed_batch.columns:
            logger.warning(f"Skipping batch: No 'trip_duration' after preprocessing (check schema)")
            continue

        for idx, row in processed_batch.iterrows():
            if "trip_duration" not in row.index or pd.isna(row["trip_duration"]):
                continue  # Skip invalid rows

            x = row.drop("trip_duration").to_dict()
            y = row["trip_duration"]
            x.pop("vendor_id")
            x.pop("rate_code")
            x.pop("payment_type")
            x.pop("pickup_longitude")
            x.pop("pickup_latitude")
            x.pop("pickup_datetime")
            x.pop("dropoff_datetime")

            y_pred = model.predict_one(x)
            mae.update(y, y_pred)
            rmse.update(y, y_pred)

            model.learn_one(x, y)

        # log each batch into mlflow # TODO is this too much?
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f'hatr_{model_params["grace_period"]}_{model_params["model_selector_decay"]}_checkpoint_{batch_count}.pkl'
        )
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(model, f)
        mlflow.log_artifact(checkpoint_path)
        mlflow.log_param("grace_period", model_params["grace_period"])
        mlflow.log_param("model_selector_decay", model_params["model_selector_decay"])
        mlflow.log_metrics({
            'mae': mae.get(),
            'rmse': rmse.get(),
            'samples_processed': data_loader.batch_size * batch_count
        }, step=data_loader.batch_size * batch_count)

        batch_count += 1

    final_model_path = os.path.join(
        checkpoint_dir,
        f'hatr_{model_params["grace_period"]}_{model_params["model_selector_decay"]}_final.pkl'
    )
    with open(final_model_path, 'wb') as f:
        pickle.dump(model, f)
    mlflow.log_artifact(final_model_path)
    mlflow.log_param("grace_period", model_params["grace_period"])
    mlflow.log_param("model_selector_decay", model_params["model_selector_decay"])
    final_metrics = {
        'final_mae': mae.get(),
        'final_rmse': rmse.get(),
    }
    mlflow.log_metrics(final_metrics)
    run_id = mlflow.active_run().info.run_id
    mlflow.end_run()
    # TODO log some plots to mlflow
    return model, final_metrics, run_id


def run_training(*args, **kwargs):
    mlflow.set_experiment("nyc_taxi_duration")
    model_params = [
        {'grace_period': gp, 'model_selector_decay': msd}
        for gp in [50, 100, 200]
        for msd in [0.3, 0.6, 0.95]
    ]

    best_model = None
    best_metrics = None
    best_run_id = None
    best_params = None

    metric_to_track = 'final_rmse'
    best_metric_value = float('inf')

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
            logger.info(f"Model not better. Best {metric_to_track}: {best_metric_value:.4f}, Current: {current_metric:.4f}")

    # register best model to MLflow Model Registry
    if best_model is not None:
        logger.info(f"\nRegistering best model with {metric_to_track}: {best_metric_value:.4f}")

        # save best model
        best_model_path = f'checkpoints/hatr_best_model.pkl'
        with open(best_model_path, 'wb') as f:
            pickle.dump(best_model, f)

        # log to the best run
        with mlflow.start_run(run_id=best_run_id):
            mlflow.log_artifact(best_model_path)
            mlflow.log_metric('is_best_model', 1)

            # register model to Model Registry
            model_uri = f"runs:/{best_run_id}/hatr_best_model.pkl"

            try:
                mlflow.register_model(
                    model_uri=model_uri,
                    name="HoeffdingAdaptiveTreeRegressor"
                )
                logger.info(f"Model registered successfully")
            except Exception as e:
                logger.info(f"Model already exists in registry, updating version: {e}")
                # if model name already exists, it will create a new version
                mlflow.register_model(
                    model_uri=model_uri,
                    name="HoeffdingAdaptiveTreeRegressor"
                )

        logger.info("\n" + "=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Best hyperparameters: {best_params}")
        logger.info(f"Best metrics:")
        for metric_name, value in best_metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
        logger.info(f"Best run ID: {best_run_id}")
        logger.info("=" * 60)

        return best_model, best_metrics, best_params

    else:
        logger.info("No models were trained successfully")
        return None, None, None
