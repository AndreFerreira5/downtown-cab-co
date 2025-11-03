"""Preprocessing module for NYC Yellow Taxi trip records."""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaxiDataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessor for NYC Yellow Taxi trip records.

    Handles data cleaning, validation, feature engineering, and preparation
    for model training.
    """

    def __init__(
            self,
            standardize_columns: bool = False,
            remove_outliers: bool = True,
            create_features: bool = False,
            target_column: str = "trip_duration",
    ):
        """
        Initialize the preprocessor.

        Args:
            remove_outliers: Whether to remove outlier records
            create_features: Whether to create engineered features
            target_column: Name of the target variable (trip duration)
        """
        self.standardize_columns = standardize_columns
        self.remove_outliers = remove_outliers
        self.create_features = create_features
        self.target_column = target_column

        # Will be set during fit
        self.vendor_mapping_ = None
        self.payment_mapping_ = None
        self.ratecode_mapping_ = None

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the preprocessor (learn mappings if needed).

        Args:
            X: Input dataframe
            y: Target variable (unused)

        Returns:
            self
        """
        # Store categorical mappings observed during training
        if "VendorID" in X.columns:
            self.vendor_mapping_ = {
                "CMT": "1", "VTS": "2",
                "1": "1", "2": "2", "6": "6", "7": "7"
            }

        if "payment_type" in X.columns:
            self.payment_mapping_ = {
                "CSH": "2", "CRD": "1", "NOC": "3", "DIS": "4",
                "0": "0", "1": "1", "2": "2", "3": "3",
                "4": "4", "5": "5", "6": "6"
            }

        if "RatecodeID" in X.columns:
            self.ratecode_mapping_ = {
                "1": "1", "2": "2", "3": "3", "4": "4",
                "5": "5", "6": "6", "99": "99"
            }

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input dataframe.

        Args:
            X: Input dataframe

        Returns:
            Preprocessed dataframe
        """
        df = X.copy()

        logger.info(f"Starting preprocessing with {len(df)} records")

        # Step 1: Handle column name variations if needed
        if self.standardize_columns:
            df = self._standardize_columns(df)

        # Step 2: Parse datetime columns
        df = self._parse_datetimes(df)

        # Step 3: Create target variable (trip duration in seconds)
        df = self._create_target(df)

        # Step 4: Handle data types
        df = self._convert_data_types(df)

        # Step 5: Handle missing values
        df = self._handle_missing_values(df)

        # Step 6: Remove invalid records
        df = self._remove_invalid_records(df)

        # Step 7: Remove outliers (optional)
        if self.remove_outliers:
            df = self._remove_outliers(df)

        # Step 8: Feature engineering (optional)
        if self.create_features:
            df = self._create_features(df)

        logger.info(f"Preprocessing complete with {len(df)} records")

        return df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to match data dictionary."""
        column_mapping = {
            "vendorid": "VendorID",
            "vendor_id": "VendorID",
            "ratecodeid": "RatecodeID",
            "rate_code_id": "RatecodeID",
            "pulocationid": "PULocationID",
            "dolocationid": "DOLocationID",
            "pu_location_id": "PULocationID",
            "do_location_id": "DOLocationID",
        }

        df.columns = df.columns.str.lower()
        df = df.rename(columns=column_mapping)

        return df

    def _parse_datetimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse pickup and dropoff datetime columns."""
        datetime_cols = ["pickup_datetime", "dropoff_datetime"]

        for col in datetime_cols:
            if col in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col], errors="coerce")

        return df

    def _create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create trip duration target variable in seconds."""
        if "pickup_datetime" in df.columns and "dropoff_datetime" in df.columns:
            df[self.target_column] = (
                    df["dropoff_datetime"] - df["pickup_datetime"]
            ).dt.total_seconds()

        return df

    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to appropriate data types."""
        # Numeric columns
        numeric_cols = [
            "passenger_count", "trip_distance", "fare_amount",
            "extra", "mta_tax", "tip_amount", "tolls_amount",
            "improvement_surcharge", "total_amount", "congestion_surcharge",
            "airport_fee", "cbd_congestion_fee",
            "pickup_longitude", "pickup_latitude",
            "dropoff_longitude", "dropoff_latitude"
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Categorical columns with mapping
        if "VendorID" in df.columns and self.vendor_mapping_:
            df["VendorID"] = df["VendorID"].astype(str).map(
                self.vendor_mapping_
            ).fillna(df["VendorID"])

        if "payment_type" in df.columns and self.payment_mapping_:
            df["payment_type"] = df["payment_type"].astype(str).map(
                self.payment_mapping_
            ).fillna(df["payment_type"])

        if "RatecodeID" in df.columns and self.ratecode_mapping_:
            df["RatecodeID"] = df["RatecodeID"].astype(str).fillna("1")

        # Store and forward flag
        if "store_and_fwd_flag" in df.columns:
            df["store_and_fwd_flag"] = df["store_and_fwd_flag"].astype(str).str.upper()
            df["store_and_fwd_flag"] = df["store_and_fwd_flag"].replace(
                {"Y": 1, "N": 0, "": 0}
            )
            df["store_and_fwd_flag"] = pd.to_numeric(
                df["store_and_fwd_flag"], errors="coerce"
            ).fillna(0).astype(int)

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with appropriate strategies."""
        # Passenger count: default to 1
        if "passenger_count" in df.columns:
            df["passenger_count"] = df["passenger_count"].fillna(1)

        # Store and forward flag: default to 0 (N)
        if "store_and_fwd_flag" in df.columns:
            df["store_and_fwd_flag"] = df["store_and_fwd_flag"].fillna(0)

        # Rate code: default to 1 (Standard rate)
        if "RatecodeID" in df.columns:
            df["RatecodeID"] = df["RatecodeID"].fillna("1")

        # Financial columns: fill with 0
        financial_cols = [
            "extra", "mta_tax", "tip_amount", "tolls_amount",
            "improvement_surcharge", "congestion_surcharge",
            "airport_fee", "cbd_congestion_fee"
        ]
        for col in financial_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        return df

    def _remove_invalid_records(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove records with invalid or impossible values."""
        initial_count = len(df)

        # Remove records with invalid datetimes
        if "pickup_datetime" in df.columns:
            df = df[df["pickup_datetime"].notna()]
        if "dropoff_datetime" in df.columns:
            df = df[df["dropoff_datetime"].notna()]

        # Remove records with negative or zero trip duration
        if self.target_column in df.columns:
            df = df[df[self.target_column] > 0]

        # Remove records with invalid passenger count
        if "passenger_count" in df.columns:
            df = df[(df["passenger_count"] >= 1) & (df["passenger_count"] <= 9)]

        # Remove records with invalid trip distance
        if "trip_distance" in df.columns:
            df = df[df["trip_distance"] > 0]

        # Remove records with invalid fare amounts
        if "fare_amount" in df.columns:
            df = df[df["fare_amount"] > 0]

        # Remove records with invalid coordinates (if present)
        if "pickup_longitude" in df.columns and "pickup_latitude" in df.columns:
            df = df[
                (df["pickup_longitude"] != 0) &
                (df["pickup_latitude"] != 0) &
                (df["pickup_longitude"].between(-180, 180)) &
                (df["pickup_latitude"].between(-90, 90))
                ]

        if "dropoff_longitude" in df.columns and "dropoff_latitude" in df.columns:
            df = df[
                (df["dropoff_longitude"] != 0) &
                (df["dropoff_latitude"] != 0) &
                (df["dropoff_longitude"].between(-180, 180)) &
                (df["dropoff_latitude"].between(-90, 90))
                ]

        logger.info(f"Removed {initial_count - len(df)} invalid records")

        return df

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outlier records based on domain knowledge and IQR."""
        initial_count = len(df)

        # Trip duration: 1 minute to 3 hours (reasonable NYC taxi trip)
        if self.target_column in df.columns:
            df = df[
                (df[self.target_column] >= 60) &
                (df[self.target_column] <= 10800)
                ]

        # Trip distance: 0.1 to 100 miles
        if "trip_distance" in df.columns:
            df = df[
                (df["trip_distance"] >= 0.1) &
                (df["trip_distance"] <= 100)
                ]

        # Fare amount: $2.50 to $500 (reasonable range)
        if "fare_amount" in df.columns:
            df = df[
                (df["fare_amount"] >= 2.5) &
                (df["fare_amount"] <= 500)
                ]

        # Speed check: 0.5 to 65 mph (reasonable NYC speeds)
        if "trip_distance" in df.columns and self.target_column in df.columns:
            df["speed_mph"] = df["trip_distance"] / (df[self.target_column] / 3600)
            df = df[(df["speed_mph"] >= 0.5) & (df["speed_mph"] <= 65)]
            df = df.drop(columns=["speed_mph"])

        logger.info(f"Removed {initial_count - len(df)} outlier records")

        return df

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features for modeling."""
        if "pickup_datetime" not in df.columns:
            return df

        # Temporal features
        df["pickup_hour"] = df["pickup_datetime"].dt.hour
        df["pickup_day"] = df["pickup_datetime"].dt.day
        df["pickup_weekday"] = df["pickup_datetime"].dt.dayofweek
        df["pickup_month"] = df["pickup_datetime"].dt.month
        df["pickup_year"] = df["pickup_datetime"].dt.year

        # Is weekend
        df["is_weekend"] = (df["pickup_weekday"] >= 5).astype(int)

        # Rush hour flags (7-10 AM and 4-7 PM on weekdays)
        df["is_morning_rush"] = (
                (df["pickup_hour"].between(7, 10)) &
                (df["pickup_weekday"] < 5)
        ).astype(int)

        df["is_evening_rush"] = (
                (df["pickup_hour"].between(16, 19)) &
                (df["pickup_weekday"] < 5)
        ).astype(int)

        # Night trip (10 PM - 6 AM)
        df["is_night"] = (
                (df["pickup_hour"] >= 22) | (df["pickup_hour"] <= 6)
        ).astype(int)

        # Distance-based features (if coordinates available)
        if all(col in df.columns for col in [
            "pickup_longitude", "pickup_latitude",
            "dropoff_longitude", "dropoff_latitude"
        ]):
            # Haversine distance
            df["haversine_distance"] = self._calculate_haversine(
                df["pickup_latitude"], df["pickup_longitude"],
                df["dropoff_latitude"], df["dropoff_longitude"]
            )

            # Manhattan distance (approximation)
            df["manhattan_distance"] = (
                    np.abs(df["pickup_latitude"] - df["dropoff_latitude"]) +
                    np.abs(df["pickup_longitude"] - df["dropoff_longitude"])
            )

            # Bearing (direction of travel)
            df["bearing"] = self._calculate_bearing(
                df["pickup_latitude"], df["pickup_longitude"],
                df["dropoff_latitude"], df["dropoff_longitude"]
            )

        # Airport trip indicator (if rate code available)
        if "RatecodeID" in df.columns:
            df["is_airport_trip"] = df["RatecodeID"].isin(["2", "3"]).astype(int)

        logger.info(
            f"Created {len([c for c in df.columns if c not in ['pickup_datetime', 'dropoff_datetime']])} features")

        return df

    @staticmethod
    def _calculate_haversine(lat1, lon1, lat2, lon2):
        """Calculate Haversine distance between two points (in miles)."""
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))

        # Radius of Earth in miles
        r = 3959.87433

        return c * r

    @staticmethod
    def _calculate_bearing(lat1, lon1, lat2, lon2):
        """Calculate bearing between two points (in degrees)."""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        dlon = lon2 - lon1

        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

        bearing = np.degrees(np.arctan2(x, y))
        bearing = (bearing + 360) % 360

        return bearing


def preprocess_taxi_data(
        df: pd.DataFrame,
        standardize_columns: bool = False,
        remove_outliers: bool = True,
        create_features: bool = False,
        target_column: str = "trip_duration",
        fit_preprocessor: bool = False,
) -> Tuple[pd.DataFrame, TaxiDataPreprocessor]:
    """
    Convenience function to preprocess taxi data.

    Args:
        df: Input dataframe
        remove_outliers: Whether to remove outliers
        create_features: Whether to create engineered features
        target_column: Name of target variable
        fit_preprocessor: Whether to fit the preprocessor

    Returns:
        Tuple of (preprocessed dataframe, fitted preprocessor)
    """
    preprocessor = TaxiDataPreprocessor(
        standardize_columns=standardize_columns,
        remove_outliers=remove_outliers,
        create_features=create_features,
        target_column=target_column,
    )

    if fit_preprocessor:
        preprocessor.fit(df)

    df_processed = preprocessor.transform(df)

    return df_processed, preprocessor


def get_feature_columns(include_target: bool = False) -> list:
    """
    Get list of feature columns for modeling.

    Args:
        include_target: Whether to include target column

    Returns:
        List of feature column names
    """
    feature_cols = [
        # Temporal features
        "pickup_hour", "pickup_day", "pickup_weekday",
        "pickup_month", "pickup_year",
        "is_weekend", "is_morning_rush", "is_evening_rush", "is_night",

        # Trip characteristics
        "passenger_count", "trip_distance",

        # Location features (if available)
        "PULocationID", "DOLocationID",
        "haversine_distance", "manhattan_distance", "bearing",

        # Categorical features
        "VendorID", "RatecodeID", "payment_type",
        "store_and_fwd_flag", "is_airport_trip",
    ]

    if include_target:
        feature_cols.append("trip_duration")

    return feature_cols
