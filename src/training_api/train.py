import mlflow
from river import tree, metrics as river_metrics
from .data.loader import DataLoader
from .data.processer import preprocess_taxi_data
import pickle
import logging
import pandas as pd
import os
import gc
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb

from collections import namedtuple


from sklearn.linear_model import Ridge

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


def plot_model_anatomy(model, daily_stats):
    """
    Visualizes the Linear Trend vs. The Seasonal Wave
    model: The trained Ridge/LinearRegression model (Model A)
    daily_stats: The dataframe used to train it (with date_int, sin_time, cos_time)
    """

    # 1. Get the Coefficients (The "Brain" of the model)
    w_slope = model.coef_[0]  # Weight for date_int
    w_sin = model.coef_[1]  # Weight for sin_time
    w_cos = model.coef_[2]  # Weight for cos_time
    intercept = model.intercept_

    print(f"Learned Slope: {w_slope:.6f} (duration change per day)")
    print(f"Learned Amplitude: {np.sqrt(w_sin ** 2 + w_cos ** 2):.2f} (seconds)")

    # 2. Reconstruct the Components
    dates = daily_stats['date_int']

    # Component A: The Pure Slope (Global Drift)
    # y = mx + b
    # We center it around the mean to make it graphable on the same chart
    drift_component = (dates * w_slope) + intercept

    # Component B: The Pure Wave (Seasonality)
    # y = A*sin + B*cos
    # We add the mean duration so it sits at the right height
    mean_val = daily_stats['trip_duration'].mean()
    seasonality_component = (daily_stats['sin_time'] * w_sin) + \
                            (daily_stats['cos_time'] * w_cos) + mean_val

    # Component C: The Full Prediction (Slope + Wave)
    full_prediction = model.predict(daily_stats[['date_int', 'sin_time', 'cos_time']])

    # 3. Plotting
    plt.figure(figsize=(15, 8))

    # A. Raw Data (The Blue Dots)
    plt.scatter(dates, daily_stats['trip_duration'],
                alpha=0.3, color='gray', s=10, label='Actual Daily Data')

    # B. The Global Drift (The Dashed Line)
    plt.plot(dates, drift_component,
             linestyle='--', color='black', linewidth=2, label='Global Drift (Linear)')

    # C. The Seasonal Pattern (The Green Wave)
    # We just plot a snippet or the whole thing.
    # Let's plot the wave relative to the drift to show the final model.
    plt.plot(dates, full_prediction,
             color='red', linewidth=3, label='Full Hybrid Model (Drift + Seasonality)')

    plt.title("Model A Deconstructed: Drift vs. Seasonality", fontsize=14)
    plt.xlabel("Time (Days)")
    plt.ylabel("Trip Duration")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig("model_a_anatomy.png")
    plt.show()

def run_training(commit_sha: str, model_name: str, experiment_name: str):
    data_loader = DataLoader("training/", batch_size=1_000_000, download_dataset=False)

    daily_stats_accumulator = []
    while (batch := data_loader.load_next_batch()) is not None and not batch.empty:
        processed_batch, _ = preprocess_taxi_data(batch, create_features=True, copy=False)
        chunk_sums = processed_batch.groupby(['date_int', 'sin_time', 'cos_time'])['trip_duration'].agg(['sum', 'count']).reset_index()
        daily_stats_accumulator.append(chunk_sums)

        del processed_batch
        gc.collect()

    print(daily_stats_accumulator)

    plot_df = pd.concat(daily_stats_accumulator)

    final_stats = plot_df.groupby(['date_int', 'sin_time', 'cos_time'])[['sum', 'count']].sum().reset_index()

    final_stats['avg_duration'] = final_stats['sum'] / final_stats['count']

    model_a = Ridge(alpha=1.0)
    X_trend = final_stats[['date_int', 'sin_time', 'cos_time']]
    y_trend = final_stats['avg_duration']
    model_a.fit(X_trend, y_trend)

    """
    plot_model_anatomy(model_a, plot_df)

    # 2. Sort by date_int to ensure the line plot is connected correctly
    plot_df = plot_df.sort_values('date_int')

    # 3. Create the Plot
    plt.figure(figsize=(14, 6))

    # Plot raw daily dots
    plt.scatter(plot_df['date_int'], plot_df['trip_duration'],
                alpha=0.4, s=15, label='Daily Average (Raw)', color='#1f77b4')

    # Plot a smooth trend line (7-day rolling average)
    # This helps visualize the 'Trend + Seasonality' signal clearly
    plt.plot(plot_df['date_int'], plot_df['trip_duration'].rolling(window=7).mean(),
             color='#d62728', linewidth=2, label='7-Day Trend')

    plt.title("Trend Analysis: Daily Average Trip Duration", fontsize=14)
    plt.xlabel("Time (Date Integer)", fontsize=12)
    plt.ylabel("Avg Duration (Minutes)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig("trend_analysis.png")
    plt.show()
    """

    with open(f"model_a.pkl", "wb") as f:
        pickle.dump(model_a, f)
    mlflow.log_artifact(f"model_a.pkl")

    booster = None
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,  # conservative learning rate for incremental
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'n_jobs': -1,
        'verbose': -1
    }

    data_loader.reset()
    while (batch := data_loader.load_next_batch()) is not None and not batch.empty:
        processed_batch, _ = preprocess_taxi_data(batch, create_features=True)
        trend_features = processed_batch[['date_int', 'sin_time', 'cos_time']]
        baseline_preds = model_a.predict(trend_features)

        y_residual = processed_batch['trip_duration'] - baseline_preds
        drop_cols = ['trip_duration', 'tpep_pickup_datetime', 'date_int', 'sin_time', 'cos_time']
        X_residual = processed_batch.drop(columns=drop_cols, errors='ignore').select_dtypes(include=['number', 'category'])

        train_set = lgb.Dataset(X_residual, y_residual, free_raw_data=True)

        booster = lgb.train(
            params,
            train_set,
            num_boost_round=50,
            init_model=booster,
            keep_training_booster=True
        )

        del processed_batch, X_residual, y_residual, train_set
        gc.collect()

    print("Training Complete. Saving Model B...")
    with open(f"model_b.pkl", "wb") as f:
        pickle.dump(booster, f)
    mlflow.log_artifact(f"model_b.pkl")



def run_training_(commit_sha: str, model_name: str, experiment_name: str):
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
