"""Preprocessing module for NYC Yellow Taxi trip records."""

import logging
from typing import Tuple, List

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
            copy: bool = True, # Copy dataframe to avoid modifying it in-place
            standardize_columns: bool = True,
            remove_outliers: bool = True,
            create_features: bool = False,
            target_column: str = "trip_duration",
            columns_to_keep: List[str] = None,
            datetime_cols: List[str] = None,
            categorical_cols: List[str] = None,
            numeric_cols: List[str] = None,
    ):
        """
        Initialize the preprocessor.

        Args:
            copy: Whether to copy the input dataframe before processing
            standardize_columns: Whether to standardize column names
            remove_outliers: Whether to remove outlier records
            create_features: Whether to create engineered features
            target_column: Name of the target variable (trip duration)
            columns_to_keep: List of columns to keep after preprocessing
            datetime_cols: List of datetime columns to parse
            categorical_cols: List of categorical columns to handle as strings
            numeric_cols: List of numeric columns to convert to numeric
        """

        self.copy = copy # For large batches, deepcopy may be expensive (double mem usage)

        self.standardize_columns = standardize_columns
        self.remove_outliers = remove_outliers
        self.create_features = create_features
        self.target_column = target_column

        self.column_mapping = {
            "vendorid": "VendorID",
            "vendor_id": "VendorID",
            "pulocationid": "PULocationID",
            "dolocationid": "DOLocationID",
            "pu_location_id": "PULocationID",
            "do_location_id": "DOLocationID",
        }

        # Default columns if none provided
        self.columns_to_keep = columns_to_keep if columns_to_keep else [
            "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
            "passenger_count", "trip_distance", "PULocationID", "DOLocationID"
        ]

        self.datetime_cols = datetime_cols if datetime_cols else [
            "tpep_pickup_datetime", "tpep_dropoff_datetime"
        ]

        self.categorical_cols = categorical_cols if categorical_cols else [
            "PULocationID", "DOLocationID"
        ]

        self.numeric_cols = numeric_cols if numeric_cols else [
            "passenger_count", "trip_distance"
        ]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input dataframe.

        Args:
            X: Input dataframe

        Returns:
            Preprocessed dataframe
        """

        # Optionally copy the dataframe to avoid modifying in-place
        df = X.copy(deep=self.copy) if self.copy else X

        logger.info(f"Starting preprocessing with {len(df)} records")

        # Step 1: Handle column name variations (optional)
        df = self._standardize_columns(df)

        # Step 2: Drop unused columns
        df = self._drop_columns(df)

        # Step 3: Parse datetime columns
        df = self._parse_datetimes(df)

        # Step 4: Remove invalid records
        df = self._remove_invalid_records(df)

        # Step 5: Remove outliers (optional)
        df = self._remove_outliers(df)

        # Step 6: Handle data types
        df = self._convert_data_types(df)

        # Step 7: Create target variable (trip duration in seconds)
        df = self._create_target(df)

        # Step 8: Handle missing values
        df = self._handle_missing_values(df)

        # Step 9: Feature engineering (optional)
        df = self._create_features(df)

        logger.info(f"Preprocessing complete with {len(df)} records")

        return df


    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to match data dictionary."""

        if not self.standardize_columns:
            return df

        # Lowercase first to make matching easier
        df.columns = df.columns.str.lower()
        # Rename using self.column_mapping
        df = df.rename(columns=self.column_mapping)

        return df

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop unnecessary columns."""

        # Log all unused columns found
        dropped_columns = [col for col in df.columns if col not in self.columns_to_keep]
        logger.info(f"Dropped {len(dropped_columns)} unused columns: {dropped_columns}")

        # Drop unused columns (even with unstardardized names)
        df = df[self.columns_to_keep]

        return df

    def _parse_datetimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse pickup and dropoff datetime columns."""

        for col in self.datetime_cols:
            if col in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col], unit='us', errors="coerce")

        return df

    def _create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create trip duration target variable in seconds."""

        if "tpep_pickup_datetime" in df.columns and "tpep_dropoff_datetime" in df.columns:
            df[self.target_column] = (
                df["dropoff_datetime"] - df["pickup_datetime"]
            ).dt.total_seconds()

        return df

    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to appropriate data types.
            Note: Dates are already parsed as datetime64 dtype.
        """

        for col in self.numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with appropriate strategies."""

        # Passenger count: default to 1
        if "passenger_count" in df.columns:
            df["passenger_count"] = df["passenger_count"].fillna(1)

        for col in self.categorical_cols:
            df[col] = df[col].fillna("0").astype(str)

        return df

    def _remove_invalid_records(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove records with invalid or impossible values."""

        initial_count = len(df)

        # Remove records with invalid datetimes
        if "tpep_pickup_datetime" in df.columns:
            df = df[df["tpep_pickup_datetime"].notna()]
        if "tpep_dropoff_datetime" in df.columns:
            df = df[df["tpep_dropoff_datetime"].notna()]

        # Remove records with negative or zero trip duration
        if self.target_column in df.columns:
            df = df[df[self.target_column] > 0]

        # Remove records with invalid passenger count
        if "passenger_count" in df.columns:
            df = df[(df["passenger_count"] >= 1) & (df["passenger_count"] <= 9)]

        # Remove records with invalid trip distance
        if "trip_distance" in df.columns:
            df = df[df["trip_distance"] > 0]

        logger.info(f"Removed {initial_count - len(df)} invalid records")

        return df

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outlier records based on domain knowledge and IQR."""

        if not self.remove_outliers:
            return df

        initial_count = len(df)

        # Trip duration: 1 minute to 3 hours (reasonable NYC taxi trip)
        if self.target_column in df.columns:
            df = df[(df[self.target_column] >= 60) & (df[self.target_column] <= 10800)]

        # Trip distance: 0.1 to 100 miles
        if "trip_distance" in df.columns:
            df = df[(df["trip_distance"] >= 0.1) & (df["trip_distance"] <= 100)]

        # Speed check: 0.5 to 65 mph (reasonable NYC speeds)
        if "trip_distance" in df.columns and self.target_column in df.columns:
            df["speed_mph"] = df["trip_distance"] / (df[self.target_column] / 3600)
            df = df[(df["speed_mph"] >= 0.5) & (df["speed_mph"] <= 65)]
            df = df.drop(columns=["speed_mph"])

        logger.info(f"Removed {initial_count - len(df)} outlier records")

        return df

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features for modeling."""

        if not self.create_features:
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
            (df["pickup_hour"].between(7, 10)) & (df["pickup_weekday"] < 5)
        ).astype(int)

        df["is_evening_rush"] = (
            (df["pickup_hour"].between(16, 19)) & (df["pickup_weekday"] < 5)
        ).astype(int)

        # Night trip (10 PM - 6 AM)
        df["is_night"] = ((df["pickup_hour"] >= 22) | (df["pickup_hour"] <= 6)).astype(
            int
        )

        logger.info(
            f"Created {len([c for c in df.columns if c not in ['pickup_datetime', 'dropoff_datetime']])} features"
        )

        return df


def preprocess_taxi_data(
        df: pd.DataFrame,
        copy: bool = True,
        standardize_columns: bool = False,
        remove_outliers: bool = True,
        create_features: bool = False,
        target_column: str = "trip_duration",
        columns_to_keep: List[str] = None,
        datetime_cols: List[str] = None,
        categorical_cols: List[str] = None,
        numeric_cols: List[str] = None,
) -> Tuple[pd.DataFrame, TaxiDataPreprocessor]:
    """
    Convenience function to preprocess taxi data.

    Args:
        :param copy: Whether to copy the input dataframe
        :param df: Input dataframe
        :param standardize_columns: Whether to standardize column names
        :param remove_outliers: Whether to remove outliers
        :param create_features: Whether to create engineered features
        :param target_column: Name of target variable
        :param columns_to_keep: List of columns to keep after preprocessing
        :param datetime_cols: List of datetime columns to parse
        :param categorical_cols: List of categorical columns to handle as strings
        :param numeric_cols: List of numeric columns to convert to numeric

    Returns:
        Tuple of (preprocessed dataframe, fitted preprocessor)
    """
    preprocessor = TaxiDataPreprocessor(
        copy=copy,
        standardize_columns=standardize_columns,
        remove_outliers=remove_outliers,
        create_features=create_features,
        target_column=target_column,
        columns_to_keep=columns_to_keep,
        datetime_cols=datetime_cols,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
    )

    df_processed = preprocessor.transform(df)

    return df_processed, preprocessor

