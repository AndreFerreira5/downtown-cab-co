import pandas as pd
import pickle
import numpy as np
import lightgbm as lgb
from .data.loader import DataLoader
from .data.processer import preprocess_taxi_data

def test_models():
    with open("model_a.pkl", "rb") as f:
        model_a = pickle.load(f)

    with open("model_b.pkl", "rb") as f:
        model_b = pickle.load(f)

    data_loader = DataLoader(data_dir="testing/", download_dataset=False, batch_size=50, years_to_download=["2013"])

    while (batch := data_loader.load_next_batch()) is not None and not batch.empty:
        processed_batch, _ = preprocess_taxi_data(batch, create_features=True, remove_outliers=True)

        real_duration = processed_batch['trip_duration'].values.tolist()

        print(real_duration)
        print(processed_batch)

        X_trend = processed_batch[['date_int', 'sin_time', 'cos_time']]
        trend_preds = model_a.predict(X_trend)

        cols_to_exclude = ['trip_duration', 'tpep_pickup_datetime', 'tpep_dropoff_datetime',
                           'date_int', 'sin_time', 'cos_time']

        X_residual = processed_batch.drop(columns=[c for c in cols_to_exclude if c in processed_batch.columns])
        X_residual = X_residual.select_dtypes(include=['number', 'category'])

        residual_preds = model_b.predict(X_residual)

        final_preds = trend_preds + residual_preds

        final_preds = np.maximum(final_preds, 0)

        print("\n" + "=" * 40)
        print("HYBRID PREDICTION REPORT")
        print("=" * 40)

        for i, (t_pred, r_pred, final) in enumerate(zip(trend_preds, residual_preds, final_preds)):
            print(f"\nTrip #{i + 1}:")
            print(f"  Date: {batch.iloc[i]['tpep_pickup_datetime']}")
            print(f"  Distance: {batch.iloc[i]['trip_distance']} miles")
            print(f"  -----------------------------")
            print(f"  [Model A] Global Trend Baseline: {t_pred:.2f} sec  ({t_pred / 60:.2f} min)")
            print(f"  [Model B] Trip Specific Adjustment: {r_pred:+.2f} sec ({r_pred / 60:+.2f} min)")
            print(f"  -----------------------------")
            print(f"  FINAL PREDICTION: {final:.2f} sec ({final / 60:.2f} min)")
            print(f"  REAL DURATION: {real_duration[i]:.2f} sec")

        exit(0)