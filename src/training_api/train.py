import mlflow
from .data.loader import DataLoader
from .data.processer import preprocess_taxi_data
import pickle
import logging
import pandas as pd
import gc
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
from .test import test_predictor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class TrendResidualModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["regressor"], "rb") as f:
            self.trend_model = pickle.load(f)
        with open(context.artifacts["booster"], "rb") as f:
            self.booster_model = pickle.load(f)

    def predict(self, context, model_input):
        model_input["tpep_pickup_datetime"] = pd.to_datetime(model_input["tpep_pickup_datetime"])
        (X, t) = preprocess_taxi_data(pd.DataFrame([model_input]), remove_outliers=False, create_features=True, predicting=True)

        trend_pred = self.model_a.predict(X[['date_int', 'sin_time', 'cos_time']])
        trend_pred = np.maximum(trend_pred, 1.0)

        X.drop(columns=['trip_duration', 'tpep_pickup_datetime', 'date_int', 'sin_time', 'cos_time'], errors='ignore')
        ratio_pred = self.model_b.predict(X)
        final_pred = trend_pred * ratio_pred

        return final_pred


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
    # y =  mx + b
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


def train_ridge_regressor(batch_sampling_perc=0.075, random_state=42, batch_size=1_000_000, download_dataset=False):
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


def train_lightgbm(regressor_model, model_params, batch_sampling_perc=0.075, random_state=42, batch_size=1_000_000,
                   download_dataset=False):
    if not model_params:
        model_params = {
            'objective': 'tweedie',
            'tweedie_variance_power': 1.15,
            'metric': 'tweedie',
            'boosting_type': 'gbdt',
            'force_col_wise': True,
            'learning_rate': 0.1,
            'num_leaves': 127,
            'min_data_in_leaf': 20,
            'feature_fraction': 0.8,
            'bagging_freq': 1,
            'bagging_fraction': 0.7,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
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

    # with open(f"model_a.pkl", "wb") as f:
    #    pickle.dump(model_a, f)
    # mlflow.log_artifact(f"model_a.pkl")

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

        # y_residual = processed_batch['trip_duration'] - baseline_preds
        y_ratio = processed_batch['trip_duration'] / baseline_preds
        drop_cols = ['trip_duration', 'tpep_pickup_datetime', 'date_int', 'sin_time', 'cos_time']
        X_residual = processed_batch.drop(columns=drop_cols, errors='ignore').select_dtypes(
            include=['number', 'category'])

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


def prepare_in_memory_data(sample_rate=0.01):
    """
    Loads, Preprocesses, and Caches data ONE time for the entire grid search.
    Returns in-memory DataFrames ready for LightGBM.
    """
    logger.info(f"--- PHASE 1: Loading & Preprocessing ({sample_rate * 100}%) ---")

    data_loader = DataLoader("training/", batch_size=500_000, download_dataset=True)
    processed_chunks = []

    # load and preprocess into RAM
    while (batch := data_loader.load_next_batch()) is not None:
        if len(batch) > 1000:
            batch = batch.sample(frac=sample_rate, random_state=42)

        # Preprocess features
        processed, _ = preprocess_taxi_data(batch, create_features=True, copy=False)
        processed_chunks.append(processed)

        del batch

    # create the master df in RAM
    full_df = pd.concat(processed_chunks, ignore_index=True)
    logger.info(f"Data Loaded. Shape: {full_df.shape}")

    # train model A (trend) once
    logger.info("--- PHASE 2: Training Trend Model (Model A) ---")

    # aggregate for ridge
    daily_stats = full_df.groupby(['date_int', 'sin_time', 'cos_time'])['trip_duration'].agg(
        ['sum', 'count']).reset_index()
    daily_stats['avg_duration'] = daily_stats['sum'] / daily_stats['count']

    model_a = Ridge(alpha=1.0)
    model_a.fit(daily_stats[['date_int', 'sin_time', 'cos_time']], daily_stats['avg_duration'])

    # calculate ratios (targets for LightGBM)
    logger.info("--- PHASE 3: Calculating Ratios ---")

    trend_features = full_df[['date_int', 'sin_time', 'cos_time']]
    baseline = model_a.predict(trend_features)
    baseline = np.maximum(baseline, 1.0)  # Safety

    y_ratio = full_df['trip_duration'] / baseline

    # prepare X for LightGBM (drop non-features)
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

        # store Truth
        y_list.append(processed['trip_duration'].values)

        # store Trend Prediction
        trend_features = processed[['date_int', 'sin_time', 'cos_time']]
        trend_val = model_a.predict(trend_features)
        trend_val = np.maximum(trend_val, 1.0)
        trend_list.append(trend_val)

        # store Features
        drop_cols = ['trip_duration', 'tpep_pickup_datetime', 'date_int', 'sin_time', 'cos_time']
        X = processed.drop(columns=drop_cols, errors='ignore').select_dtypes(include=['number', 'category'])
        X_list.append(X)

    X_val = pd.concat(X_list, ignore_index=True)
    y_val = np.concatenate(y_list)
    trend_val = np.concatenate(trend_list)

    return X_val, y_val, trend_val


def run_hyperparameter_tuning(commit_sha, model_name):
    regressor, X_train, y_train = prepare_in_memory_data(sample_rate=0.01)
    X_val, y_val_true, val_trend = get_validation_data(regressor, sample_rate=0.05)

    model_params_grid = [
        {
            'objective': 'tweedie',
            'tweedie_variance_power': tvp,
            'metric': 'tweedie',
            'boosting_type': 'gbdt',
            'force_col_wise': True,
            'learning_rate': lr,
            'num_leaves': 127,
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

    for i, params in enumerate(model_params_grid):
        logger.info(
            f"Testing Config {i + 1}/{len(model_params_grid)}: Power={params['tweedie_variance_power']}, LearningRate={params['learning_rate']}")

        mlflow.start_run()

        # create dataset object (zero-copy)
        train_set = lgb.Dataset(X_train, y_train, free_raw_data=False)

        # train (fast because data is in RAM)
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=100,
        )

        # predict on validation
        ratio_pred = booster.predict(X_val)
        final_pred = val_trend * ratio_pred  # Re-combine

        # score
        rmse = np.sqrt(mean_squared_error(y_val_true, final_pred))
        mae = mean_absolute_error(y_val_true, final_pred)

        regressor_path = f"regressor_{commit_sha}.pkl"
        booster_path = f"booster_{commit_sha}.pkl"

        with open(regressor_path, "wb") as f:
            pickle.dump(regressor, f)

        with open(booster_path, "wb") as f:
            pickle.dump(booster, f)

        artifacts = {
            "regressor": regressor_path,
            "booster": booster_path
        }

        mlflow.pyfunc.log_model(
            artifact_path="trend_residual_model",
            python_model=TrendResidualModel(),
            artifacts=artifacts,
            registered_model_name=model_name
        )

        for key in params:
            mlflow.log_param(key, params[key])

        mlflow.log_metrics(
            {"rmse": rmse, "mae": mae}
        )

        logger.info(f"--> Result: RMSE={rmse:.4f}, MAE={mae:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params
            logger.info("   *** NEW BEST ***")

        mlflow.end_run()

        # cleanup
        del train_set, booster
        gc.collect()

    return best_params


def run_training(model_params, commit_sha, model_name):
    mlflow.start_run()

    regressor = train_ridge_regressor(batch_sampling_perc=0.1, random_state=124961, download_dataset=False)
    booster = train_lightgbm(regressor_model=regressor, model_params=model_params, batch_sampling_perc=0.1,
                             random_state=124961, download_dataset=False)

    rmse, mae, r2, fig = test_predictor(regressor, booster)

    regressor_path = f"regressor_{commit_sha}_final.pkl"
    booster_path = f"booster_{commit_sha}_final.pkl"

    with open(regressor_path, "wb") as f:
        pickle.dump(regressor, f)

    with open(booster_path, "wb") as f:
        pickle.dump(booster, f)

    artifacts = {
        "regressor": regressor_path,
        "booster": booster_path
    }

    model_info = mlflow.pyfunc.log_model(
        artifact_path="trend_residual_model",
        python_model=TrendResidualModel(),
        artifacts=artifacts,
        registered_model_name=model_name
    )

    for key in model_params:
        mlflow.log_param(key, model_params[key])
    mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
    mlflow.log_figure(fig, "model_performance_dashboard.png")

    client = MlflowClient()
    #model_version = model_info.registered_model_version

    client.set_registered_model_alias(
        name=model_name,
        alias=commit_sha,
        #version=model_version
    )

    client.set_registered_model_alias(
        name=model_name,
        alias="staging",
        #version=model_version
    )

    mlflow.end_run()

    return regressor, booster
