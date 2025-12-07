import pandas as pd
import pickle
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .data.loader import DataLoader
from .data.processer import preprocess_taxi_data


def test_predictor(trend_model, booster_model, download_dataset=False):
    data_loader = DataLoader(data_dir="testing/", download_dataset=download_dataset, batch_size=50_000, years_to_download=["2013"])

    y_true = []
    y_pred = []

    print("--- Running Inference ---")
    batches_to_process = 1_000

    batch_count = 0
    while (batch := data_loader.load_next_batch()) is not None and not batch.empty:
        if batch_count >= batches_to_process:
            break
        batch = batch.sample(frac=0.075, random_state=42)

        # 1. Preprocess (Must remove outliers to compare apples-to-apples)
        processed_batch, _ = preprocess_taxi_data(batch, create_features=True, remove_outliers=True)

        if processed_batch.empty: continue

        # 2. Model A (Trend)
        # Note: Model A predicts RAW seconds (Linearly)
        X_trend = processed_batch[['date_int', 'sin_time', 'cos_time']]
        trend = trend_model.predict(X_trend)
        trend = np.maximum(trend, 1.0)


        # Model B (Residuals)
        cols_to_exclude = ['trip_duration', 'tpep_pickup_datetime', 'tpep_dropoff_datetime',
                           'date_int']
        X_residual = processed_batch.drop(columns=[c for c in cols_to_exclude if c in processed_batch.columns])
        X_residual = X_residual.select_dtypes(include=['number', 'category'])

        resid = booster_model.predict(X_residual)

        final_sec = trend * resid

        # 5. Store for Plotting
        y_true.extend(processed_batch['trip_duration'].values)
        y_pred.extend(final_sec)

        batch_count += 1
        print(f"Processed batch {batch_count}/{batches_to_process}")

    # Convert to arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Cap predictions to avoid graph explosion (e.g. max 2 hours)
    y_pred = np.minimum(y_pred, 7200)

    # --- METRICS CALCULATION ---
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("\n" + "=" * 40)
    print(f"FINAL METRICS (N={len(y_true)})")
    print("=" * 40)
    print(f"RMSE: {rmse:.2f} seconds")
    print(f"MAE:  {mae:.2f} seconds")
    print(f"R²:   {r2:.4f}")
    print("=" * 40)

    fig = plot_performance(y_true, y_pred)

    return rmse, mae, r2, fig


def plot_performance(y_true, y_pred):
    """Generates a 4-panel dashboard of model performance"""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Actual vs Predicted (Hexbin for density)
    # Good for seeing where the mass of data lies
    hb = axes[0, 0].hexbin(y_true, y_pred, gridsize=50, cmap='Blues', mincnt=1)
    axes[0, 0].plot([0, 3000], [0, 3000], 'r--', lw=2)  # Diagonal line
    axes[0, 0].set_title("Actual vs Predicted (Density)")
    axes[0, 0].set_xlabel("Actual Duration (sec)")
    axes[0, 0].set_ylabel("Predicted Duration (sec)")
    axes[0, 0].set_xlim(0, 3000)
    axes[0, 0].set_ylim(0, 3000)
    cb = fig.colorbar(hb, ax=axes[0, 0])
    cb.set_label('Count')

    # Plot 2: Residual Plot
    # Shows if errors are random (Good) or patterned (Bad)
    residuals = y_true - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.1, s=2, color='purple')
    axes[0, 1].axhline(0, color='black', linestyle='--')
    axes[0, 1].set_title("Residuals (Actual - Pred)")
    axes[0, 1].set_xlabel("Predicted Value")
    axes[0, 1].set_ylabel("Error (Seconds)")
    axes[0, 1].set_xlim(0, 3000)
    axes[0, 1].set_ylim(-1000, 1000)

    # Plot 3: Error Distribution
    # Should be a bell curve centered at 0
    sns.histplot(residuals, bins=100, kde=True, ax=axes[1, 0], color='green')
    axes[1, 0].set_title("Error Distribution")
    axes[1, 0].set_xlabel("Error (Seconds)")
    axes[1, 0].set_xlim(-1000, 1000)

    # Plot 4: Zoomed Time Series (First 100 samples)
    # Good for visual sanity check
    subset_n = 35
    x_range = range(subset_n)
    axes[1, 1].plot(x_range, y_true[:subset_n], label="Actual", color='gray', alpha=0.7)
    axes[1, 1].plot(x_range, y_pred[:subset_n], label="Predicted", color='blue', linewidth=2)
    axes[1, 1].set_title(f"First {subset_n} Trips: Detailed View")
    axes[1, 1].set_xlabel("Trip Index")
    axes[1, 1].set_ylabel("Duration (sec)")
    axes[1, 1].legend()

    plt.tight_layout()
    #plt.savefig("model_performance_dashboard.png")
    print("Graphs saved to 'model_performance_dashboard.png'")
    #plt.show()

    return fig
