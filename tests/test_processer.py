import pandas as pd
import numpy as np
import pytest
from src.training_api.data.processer import TaxiDataPreprocessor


@pytest.fixture
def sample_raw_data():
    """
    Generates a small dataframe mimicking the raw NYC taxi data.

    Timestamps are in Microseconds (16 digits) representing early 2013.
    Example: 1356998400000000 = 2013-01-01 00:00:00 UTC
    """
    # 2013-01-01 00:10:00
    base_time = 1356999000000000

    data = {
        "VendorID": [1, 2, 1, 2, 1],
        "tpep_pickup_datetime": [
            base_time,  # Valid
            base_time + 600000000,  # +10 mins
            base_time + 1200000000,  # +20 mins
            base_time + 1800000000,  # +30 mins
            base_time  # Valid
        ],
        "tpep_dropoff_datetime": [
            base_time + 1200000000,  # +20 mins (Duration: 1200s)
            base_time + 900000000,  # +15 mins from start (Duration: 5 mins = 300s)
            base_time + 1200005000,  # +20 mins + 5ms (Duration: ~0s -> Outlier!)
            base_time + 1800000000 + (4 * 3600 * 1000000),  # +4 hours (Outlier > 3h)
            base_time + 2400000000  # +40 mins
        ],
        "passenger_count": [1, 2, np.nan, 1, 0],  # NaN -> fill 1, 0 -> Invalid/Drop
        "trip_distance": [2.5, 3.0, 1.5, 10.0, 5.0],
        "PULocationID": [100, 101, 102, 103, 104],
        "DOLocationID": [200, 201, 202, 203, 204],
        # Simulate an extra column that should be dropped
        "payment_type": [1, 1, 1, 1, 1]
    }
    return pd.DataFrame(data)


def test_basic_pipeline_execution(sample_raw_data):
    """Test that the pipeline runs without errors on valid data."""
    processor = TaxiDataPreprocessor(remove_outliers=False)
    df_processed = processor.transform(sample_raw_data)

    assert not df_processed.empty
    assert "trip_duration" in df_processed.columns
    # Ensure the extra column was dropped
    assert "payment_type" not in df_processed.columns


def test_date_parsing(sample_raw_data):
    """
    Test that integer microseconds are correctly parsed to datetime objects.
    Verifies that 1356999000000000 becomes a 2013 date.
    """
    processor = TaxiDataPreprocessor(create_additional_features=True)
    # We access the protected method to test isolation, or check output
    df_time_parsed = processor._standardize_columns(sample_raw_data)
    df_time_parsed = processor._drop_columns(df_time_parsed)
    df_time_parsed = processor._parse_datetimes(df_time_parsed)

    # Check initial date parsing
    assert pd.api.types.is_datetime64_any_dtype(df_time_parsed["tpep_pickup_datetime"])

    df_time_parsed = processor._create_features(df_time_parsed)

    # Check specifically for the year 2013
    assert df_time_parsed["pickup_year"][1] == 2013


def test_trip_duration_calculation(sample_raw_data):
    """Test that trip_duration is calculated correctly in seconds."""
    processor = TaxiDataPreprocessor(remove_outliers=False)
    df = processor.transform(sample_raw_data)

    # Row 0: Dropoff (base + 20m) - Pickup (base) = 20 minutes = 1200 seconds
    # Allow small tolerance for float math
    duration = df.iloc[0]["trip_duration"]
    assert abs(duration - 1200.0) < 0.1


def test_outlier_removal(sample_raw_data):
    """Test valid removal of short/long trips and invalid passengers."""
    processor = TaxiDataPreprocessor(remove_outliers=True)
    df = processor.transform(sample_raw_data)

    # Record 0: 20 mins, 1 pax -> KEEP
    # Record 1: 5 mins, 2 pax -> KEEP
    # Record 2: ~0s duration -> DROP (Too short)
    # Record 3: 4 hours -> DROP (Too long > 3h)
    # Record 4: 0 passengers -> KEEP (Defaults to 1)

    # Indices are reset in standard pandas operations if not specified,
    # let's just check the count.
    assert len(df) == 3

    # Validate remaining durations are within bounds [60, 10800]
    assert df["trip_duration"].min() >= 60
    assert df["trip_duration"].max() <= 10800


def test_missing_values_handling(sample_raw_data):
    """Test that NaNs in passenger_count are filled with 1."""
    processor = TaxiDataPreprocessor(remove_outliers=False)
    df = processor.transform(sample_raw_data)

    # The 3rd record (index 2) had NaN passenger_count
    # In the processed df, it might have shifted index if we dropped invalid rows.
    # Let's find the record with trip_distance 1.5 (which was the NaN one)
    target_row = df[df["trip_distance"] == 1.5]

    assert not target_row.empty
    assert target_row.iloc[0]["passenger_count"] == 1.0


def test_feature_engineering(sample_raw_data):
    """Test creation of time-based features."""
    processor = TaxiDataPreprocessor(create_additional_features=True, remove_outliers=False)
    df = processor.transform(sample_raw_data)

    expected_features = [
        "pickup_hour",
        "pickup_day",
        "pickup_month",
        "pickup_year",
        "is_weekend"
    ]

    for feat in expected_features:
        assert feat in df.columns

    # Check specific values
    # base_time is roughly 2013-01-01 (Tuesday)
    assert df.iloc[0]["pickup_year"] == 2013
    assert df.iloc[0]["is_weekend"] == 0  # Tuesday is not weekend