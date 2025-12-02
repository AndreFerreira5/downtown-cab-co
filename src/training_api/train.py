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
from .test import test_predictor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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



def train_ridge_regressor(batch_sampling_perc=0.075, random_state=42, batch_size=1_000_000):
    data_loader = DataLoader("training/", batch_size=batch_size, download_dataset=False)
    daily_stats_accumulator = []
    while (batch := data_loader.load_next_batch()) is not None and not batch.empty:
        if len(batch) > 1000:
            batch = batch.sample(frac=batch_sampling_perc, random_state=random_state)
        processed_batch, _ = preprocess_taxi_data(batch, create_features=True, copy=False)
        chunk_sums = processed_batch.groupby(['date_int', 'sin_time', 'cos_time'])['trip_duration'].agg(
            ['sum', 'count']).reset_index()
        daily_stats_accumulator.append(chunk_sums)

        del processed_batch
        gc.collect()

    plot_df = pd.concat(daily_stats_accumulator)
    final_stats = plot_df.groupby(['date_int', 'sin_time', 'cos_time'])[['sum', 'count']].sum().reset_index()
    final_stats['avg_duration'] = final_stats['sum'] / final_stats['count']

    regressor = Ridge(alpha=1.0)
    X_trend = final_stats[['date_int', 'sin_time', 'cos_time']]
    y_trend = final_stats['avg_duration']
    regressor.fit(X_trend, y_trend)
    return regressor

def train_lightgbm(regressor_model, model_params, batch_sampling_perc=0.075, random_state=42, batch_size=1_000_000):
    if not model_params:
        model_params = {
            'objective': 'tweedie',
            'tweedie_variance_power': 1.6,
            'metric': 'tweedie',
            'boosting_type': 'gbdt',
            'force_col_wise': True,
            'learning_rate': 0.1,
            'num_leaves': 63,
            'min_data_in_leaf': 50,
            'feature_fraction': 0.8,
            'bagging_freq': 1,
            'bagging_fraction': 0.7,
            'lambda_l1': 2.0,
            'lambda_l2': 2.0,
            'n_jobs': -1,
            'verbose': -1
        }

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

    #with open(f"model_a.pkl", "wb") as f:
    #    pickle.dump(model_a, f)
    #mlflow.log_artifact(f"model_a.pkl")

    booster = None
    data_loader = DataLoader("training/", batch_size=batch_size, download_dataset=False)
    data_loader.reset()
    while (batch := data_loader.load_next_batch()) is not None and not batch.empty:
        if len(batch) > 1000:
            batch = batch.sample(frac=batch_sampling_perc, random_state=random_state)
        processed_batch, _ = preprocess_taxi_data(batch, create_features=True)
        trend_features = processed_batch[['date_int', 'sin_time', 'cos_time']]
        baseline_preds = regressor_model.predict(trend_features)

        # safety clamp, force baseline to be at least 1 second
        baseline_preds = np.maximum(baseline_preds, 1.0)

        #y_residual = processed_batch['trip_duration'] - baseline_preds
        y_ratio = processed_batch['trip_duration'] / baseline_preds
        drop_cols = ['trip_duration', 'tpep_pickup_datetime', 'date_int', 'sin_time', 'cos_time']
        X_residual = processed_batch.drop(columns=drop_cols, errors='ignore').select_dtypes(include=['number', 'category'])

        train_set = lgb.Dataset(X_residual, y_ratio, free_raw_data=True)

        booster = lgb.train(
            model_params,
            train_set,
            num_boost_round=3,
            init_model=booster,
            keep_training_booster=True
        )

        del processed_batch, X_residual, y_ratio, train_set
        gc.collect()

    return booster

    #print("Training Complete. Saving Model B...")
    #with open(f"model_b.pkl", "wb") as f:
    #    pickle.dump(booster, f)
    #mlflow.log_artifact(f"model_b.pkl")


def prepare_in_memory_data(sample_rate=0.01):
    """
    Loads, Preprocesses, and Caches data ONE time for the entire grid search.
    Returns in-memory DataFrames ready for LightGBM.
    """
    logger.info(f"--- PHASE 1: Loading & Preprocessing ({sample_rate * 100}%) ---")

    data_loader = DataLoader("training/", batch_size=500_000, download_dataset=True)
    processed_chunks = []

    # 1. Load and Preprocess into RAM
    while (batch := data_loader.load_next_batch()) is not None:
        if len(batch) > 1000:
            batch = batch.sample(frac=sample_rate, random_state=42)

        # Preprocess features
        processed, _ = preprocess_taxi_data(batch, create_features=True, copy=False)
        processed_chunks.append(processed)

        del batch

    # Create the Master DataFrame (In RAM)
    # 1% of 300M rows = 3M rows. ~300MB RAM. Very safe.
    full_df = pd.concat(processed_chunks, ignore_index=True)
    logger.info(f"Data Loaded. Shape: {full_df.shape}")

    # 2. Train Model A (Trend) ONCE
    logger.info("--- PHASE 2: Training Trend Model (Model A) ---")

    # Aggregate for Ridge
    daily_stats = full_df.groupby(['date_int', 'sin_time', 'cos_time'])['trip_duration'].agg(
        ['sum', 'count']).reset_index()
    daily_stats['avg_duration'] = daily_stats['sum'] / daily_stats['count']

    model_a = Ridge(alpha=1.0)
    model_a.fit(daily_stats[['date_int', 'sin_time', 'cos_time']], daily_stats['avg_duration'])

    # 3. Calculate Ratios (Targets for LightGBM)
    logger.info("--- PHASE 3: Calculating Ratios ---")

    trend_features = full_df[['date_int', 'sin_time', 'cos_time']]
    baseline = model_a.predict(trend_features)
    baseline = np.maximum(baseline, 1.0)  # Safety

    y_ratio = full_df['trip_duration'] / baseline

    # Prepare X for LightGBM (Drop non-features)
    drop_cols = ['trip_duration', 'tpep_pickup_datetime', 'date_int', 'sin_time', 'cos_time']
    X_train = full_df.drop(columns=drop_cols, errors='ignore').select_dtypes(include=['number', 'category'])

    return model_a, X_train, y_ratio


def get_validation_data(model_a, sample_rate=0.05):
    """
    Loads 2013 validation data into RAM for fast scoring.
    """
    logger.info("--- PHASE 4: Loading Validation Data ---")
    data_loader = DataLoader("testing/", batch_size=500_000, years_to_download=["2013"], download_dataset=True)

    X_list = []
    y_list = []
    trend_list = []

    while (batch := data_loader.load_next_batch()) is not None:
        if len(batch) > 1000:
            batch = batch.sample(frac=sample_rate, random_state=55)

        processed, _ = preprocess_taxi_data(batch, create_features=True, remove_outliers=True)
        if processed.empty: continue

        # Store Truth
        y_list.append(processed['trip_duration'].values)

        # Store Trend Prediction
        trend_features = processed[['date_int', 'sin_time', 'cos_time']]
        trend_val = model_a.predict(trend_features)
        trend_val = np.maximum(trend_val, 1.0)
        trend_list.append(trend_val)

        # Store Features
        drop_cols = ['trip_duration', 'tpep_pickup_datetime', 'date_int', 'sin_time', 'cos_time']
        X = processed.drop(columns=drop_cols, errors='ignore').select_dtypes(include=['number', 'category'])
        X_list.append(X)

    X_val = pd.concat(X_list, ignore_index=True)
    y_val = np.concatenate(y_list)
    trend_val = np.concatenate(trend_list)

    return X_val, y_val, trend_val


def run_hyperparameter_tuning():
    #mlflow.set_experiment(experiment_name)

    model_a, X_train, y_train = prepare_in_memory_data(sample_rate=0.01)
    # 2. PREPARE VALIDATION ONCE
    X_val, y_val_true, val_trend = get_validation_data(model_a, sample_rate=0.05)


    model_params_grid = [
        {
        'objective': 'tweedie',
        'tweedie_variance_power': tvp,
        'metric': 'tweedie',
        'boosting_type': 'gbdt',
        'force_col_wise': True,
        'learning_rate': lr,
        'num_leaves': 255,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.8,
        'bagging_freq': 1,
        'bagging_fraction': 0.7,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'n_jobs': -1,
        'verbose': -1
        }
        for tvp in [1.15, 1.5, 1.75]
        for lr in [0.01, 0.05, 0.1]
    ]

    best_rmse = float('inf')
    best_params = None

    logger.info(f"Starting Grid Search with {len(model_params_grid)} combinations...")

    # 4. FAST LOOP (In-Memory)
    for i, params in enumerate(model_params_grid):
        logger.info(
            f"Testing Config {i + 1}/{len(model_params_grid)}: Power={params['tweedie_variance_power']}, LearningRate={params['learning_rate']}")

        # Create Dataset Object (Zero-copy)
        train_set = lgb.Dataset(X_train, y_train, free_raw_data=False)

        # Train (Fast because data is in RAM)
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=100,
        )

        # Predict on Validation
        ratio_pred = booster.predict(X_val)
        final_pred = val_trend * ratio_pred  # Re-combine

        # Score
        rmse = np.sqrt(mean_squared_error(y_val_true, final_pred))
        mae = mean_absolute_error(y_val_true, final_pred)

        logger.info(f"--> Result: RMSE={rmse:.4f}, MAE={mae:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params
            logger.info("   *** NEW BEST ***")

        # Cleanup
        del train_set, booster
        gc.collect()

    exit(0)

    best_models = None
    best_metrics = None
    best_run_id = None
    best_params = None

    #metric_to_track = "final_rmse"
    best_rmse_value = float("inf")

    regressor = train_ridge_regressor(batch_sampling_perc=0.01)
    # train all model combinations
    for mp in model_params:
        logger.info(f"Training with params: {mp}")
        #model, final_metrics, run_id = train_lightgbm(mp)
        booster_model = train_lightgbm(regressor_model=regressor, model_params=mp, batch_sampling_perc=0.01)
        current_rmse, current_mae, current_r2 = test_predictor(regressor, booster_model)

        # compare and track best model
        #current_metric = final_metrics[metric_to_track]

        if current_rmse < best_rmse_value:
            logger.info(f"New best model! RMSE: {current_rmse:.4f}")
            best_metric_value = current_rmse
            best_models = [regressor, booster_model]
            best_metrics = [current_rmse, current_mae, current_r2]
            #best_run_id = run_id
            best_params = mp
        else:
            logger.info(
                f"Model not better. Best RMSE: {best_rmse_value:.4f}, Current: {current_rmse:.4f}"
            )

    logger.info("best params!: ", best_params)
    exit(0)

    # register best model to MLflow Model Registry
    if best_models is not None:
        logger.info(
            f"\nRegistering best model with RMSE: {best_metric_value:.4f}"
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
