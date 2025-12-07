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
        input_len = len(model_input)
        model_input["tpep_pickup_datetime"] = pd.to_datetime(model_input["tpep_pickup_datetime"])
        X, _ = preprocess_taxi_data(
            pd.DataFrame(model_input),
            remove_outliers=False,
            create_features=True,
            skip_validation=True
        )

        if len(X) != input_len: logger.warning(f"Input length {input_len} differs from processed data length {len(X)}!!")

        trend_pred = self.trend_model.predict(X[['date_int', 'sin_time', 'cos_time']])
        trend_pred = np.expm1(trend_pred)  # inverse of log1p
        trend_pred = np.maximum(trend_pred, 1.0)

        X.drop(columns=['trip_duration', 'tpep_pickup_datetime', 'date_int'], inplace=True, errors='ignore')
        X_residual = X.select_dtypes(include=['number', 'category'])
        log_correction = self.booster_model.predict(X_residual)
        ratio_multiplier = np.exp(log_correction)

        final_pred = trend_pred * ratio_multiplier

        if len(final_pred) != input_len: logger.warning(f"Input length {input_len} differs from predicted data length {len(X)}!!")

        return final_pred/60


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

        # TODO commenting these optimizations to rule out any issues related to the final training artifacts not being logged to mlflow
        #del processed_batch
        #gc.collect()

    plot_df = pd.concat(daily_stats_accumulator)
    final_stats = plot_df.groupby(['date_int', 'sin_time', 'cos_time'])[['sum', 'count']].sum().reset_index()
    final_stats['avg_duration'] = final_stats['sum'] / final_stats['count']

    regressor = Ridge(alpha=100.0)
    X_trend = final_stats[['date_int', 'sin_time', 'cos_time']]
    y_trend = np.log1p(final_stats['avg_duration'])
    regressor.fit(X_trend, y_trend)
    return regressor


def train_lightgbm(regressor_model, model_params, batch_sampling_perc=0.075, random_state=42, batch_size=1_000_000,
                   download_dataset=False):
    if not model_params:
        model_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'force_col_wise': True,
            'learning_rate': 0.1,
            'num_leaves': 127,
            'min_data_in_leaf': 20,
            'feature_fraction': 0.8,
            'bagging_freq': 1,
            'bagging_fraction': 0.7,
            'lambda_l1': 1.0,
            'lambda_l2': 1.0,
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
    batch_count = 1
    while (batch := data_loader.load_next_batch()) is not None and not batch.empty:
        #if len(batch) > 1000:
        #    batch = batch.sample(frac=batch_sampling_perc, random_state=random_state)
        processed_batch, _ = preprocess_taxi_data(batch, create_features=True)
        if processed_batch.empty: continue

        trend_features = processed_batch[['date_int', 'sin_time', 'cos_time']]
        baseline_preds_full = regressor_model.predict(trend_features)
        baseline_preds_full = np.expm1(baseline_preds_full)  # inverse of log1p
        baseline_preds_full = np.maximum(baseline_preds_full, 1.0)

        # calculate error to identify hard cases
        actual = processed_batch['trip_duration']
        # error metric, absolute percentage error
        error_ratio = np.abs(actual - baseline_preds_full) / baseline_preds_full

        # create masks for smart sampling
        # hard mask, error bigger than 20%, keep them all
        mask_hard = error_ratio > 0.20
        # easy mask, error less than 20%, keep only 5% of them
        random_mask = np.random.rand(len(processed_batch)) < 0.05
        mask_easy = (~mask_hard) & random_mask

        # combine masks
        final_mask = mask_hard | mask_easy

        # apply mask to data and baseline
        # this ensures X, y, and baseline are all perfectly aligned
        training_chunk = processed_batch[final_mask].copy()
        baseline_chunk = baseline_preds_full[final_mask]

        if len(training_chunk) == 0: continue  # Skip if empty

        # calculate targets for the chunk only
        # target = log(actual) - log(baseline)
        y_ratio_chunk = np.log1p(training_chunk['trip_duration']) - np.log1p(baseline_chunk)
        drop_cols = ['trip_duration', 'tpep_pickup_datetime', 'date_int']
        X_residual_chunk = training_chunk.drop(columns=drop_cols, errors='ignore').select_dtypes(
            include=['number', 'category'])

        if booster is not None and (batch_count % 10 == 0):
            # Predict
            pred_log_ratio = booster.predict(X_residual_chunk)
            # Reconstruct
            pred_final = baseline_chunk * np.exp(pred_log_ratio)
            actual_chunk = training_chunk['trip_duration']

            # Score
            batch_rmse = np.sqrt(mean_squared_error(actual_chunk, pred_final))
            batch_mae = mean_absolute_error(actual_chunk, pred_final)

            # Log to MLflow
            mlflow.log_metric("train_batch_rmse", batch_rmse, step=batch_count*batch_size)
            mlflow.log_metric("train_batch_mae", batch_mae, step=batch_count*batch_size)

            logger.info(f"Batch {batch_count}: RMSE={batch_rmse:.2f}, MAE={batch_mae:.2f}")

        train_set = lgb.Dataset(X_residual_chunk, y_ratio_chunk, free_raw_data=True)

        booster = lgb.train(
            model_params,
            train_set,
            num_boost_round=10,
            init_model=booster,
            keep_training_booster=True
        )

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

    model_a = Ridge(alpha=100.0)
    y_trend_log = np.log1p(daily_stats['avg_duration'])
    model_a.fit(daily_stats[['date_int', 'sin_time', 'cos_time']], y_trend_log)

    # calculate ratios (targets for LightGBM)
    logger.info("--- PHASE 3: Calculating Ratios ---")

    trend_features = full_df[['date_int', 'sin_time', 'cos_time']]
    baseline_log = model_a.predict(trend_features)

    baseline_seconds = np.expm1(baseline_log)
    baseline_seconds = np.maximum(baseline_seconds, 1.0)

    #y_ratio = full_df['trip_duration'] / baseline
    y_ratio = np.log1p(full_df['trip_duration']) - np.log1p(baseline_seconds)

    # prepare X for LightGBM (drop non-features)
    drop_cols = ['trip_duration', 'tpep_pickup_datetime', 'date_int']
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
        trend_val = np.expm1(trend_val)  # inverse of log1p
        trend_val = np.maximum(trend_val, 1.0)
        trend_list.append(trend_val)

        # store Features
        drop_cols = ['trip_duration', 'tpep_pickup_datetime', 'date_int']
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
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'force_col_wise': True,
            'learning_rate': lr,
            'num_leaves': nl,
            'min_data_in_leaf': mdil,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.7,
            'bagging_freq': 1,
            'lambda_l1': 1.0,
            'lambda_l2': 1.0,
            'n_estimators': 500,
            'n_jobs': -1,
            'verbose': -1
        }
        for lr in [0.03, 0.05]
        for nl in [127, 255]
        for mdil in [100, 300]
    ]

    best_rmse = float('inf')
    best_params = None

    logger.info(f"Starting Grid Search with {len(model_params_grid)} combinations...")

    for i, params in enumerate(model_params_grid):
        logger.info(
            f"Testing Config {i + 1}/{len(model_params_grid)}: LearningRate={params['learning_rate']}, NumberLeaves={params['num_leaves']}, MinDataInLeaf={params['min_data_in_leaf']}")

        mlflow.start_run()

        # create dataset object (zero-copy)
        train_set = lgb.Dataset(X_train, y_train, free_raw_data=False)

        y_val_log_ratio = np.log1p(y_val_true) - np.log1p(val_trend)
        valid_set = lgb.Dataset(X_val, y_val_log_ratio, reference=train_set, free_raw_data=False)

        # train (fast because data is in RAM)
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=2000,
            valid_sets=[train_set, valid_set],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),  # stop if valid score doesn't improve for 50 rounds
                lgb.log_evaluation(period=100)  # print progress every 100 rounds
            ]
        )

        # predict on validation
        ratio_pred = booster.predict(X_val, num_iteration=booster.best_iteration)
        final_pred = val_trend * np.exp(ratio_pred)

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

    rmse, mae, r2, fig = test_predictor(regressor, booster)

    for key in model_params:
        mlflow.log_param(key, model_params[key])
    mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
    mlflow.log_figure(fig, "model_performance_dashboard.png")

    client = MlflowClient()
    model_version = model_info.registered_model_version

    client.set_registered_model_alias(
        name=model_name,
        alias=commit_sha,
        version=model_version
    )

    client.set_registered_model_alias(
        name=model_name,
        alias="staging",
        version=model_version
    )

    mlflow.end_run()

    return regressor, booster
