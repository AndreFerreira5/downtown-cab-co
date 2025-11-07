"""Unit tests for data processer module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.training_api.data.processer import (
    TaxiDataPreprocessor,
    preprocess_taxi_data,
    get_feature_columns,
)


@pytest.fixture
def sample_taxi_data():
    """Create sample taxi data for testing."""
    base_time = datetime(2010, 1, 1, 10, 0, 0)
    return pd.DataFrame({
        'VendorID': ['1', '2', '1', '2'],
        'pickup_datetime': [base_time + timedelta(hours=i) for i in range(4)],
        'dropoff_datetime': [base_time + timedelta(hours=i, minutes=30) for i in range(4)],
        'passenger_count': [1, 2, 1, 3],
        'trip_distance': [1.5, 2.5, 3.0, 1.0],
        'pickup_longitude': [-73.98, -73.97, -73.99, -73.96],
        'pickup_latitude': [40.75, 40.76, 40.74, 40.77],
        'dropoff_longitude': [-73.97, -73.96, -73.98, -73.95],
        'dropoff_latitude': [40.76, 40.77, 40.75, 40.78],
        'RatecodeID': ['1', '1', '2', '1'],
        'store_and_fwd_flag': ['N', 'N', 'Y', 'N'],
        'payment_type': ['1', '2', '1', '1'],
        'fare_amount': [10.0, 15.0, 20.0, 8.0],
        'extra': [0.5, 0.5, 0.0, 0.5],
        'mta_tax': [0.5, 0.5, 0.5, 0.5],
        'tip_amount': [2.0, 0.0, 3.0, 1.5],
        'tolls_amount': [0.0, 5.0, 0.0, 0.0],
        'improvement_surcharge': [0.3, 0.3, 0.3, 0.3],
        'total_amount': [13.3, 21.3, 23.8, 10.8],
    })


@pytest.fixture
def sample_taxi_data_with_outliers():
    """Create sample taxi data with outliers."""
    base_time = datetime(2010, 1, 1, 10, 0, 0)
    return pd.DataFrame({
        'VendorID': ['1', '2', '1', '2', '1'],
        'pickup_datetime': [base_time + timedelta(hours=i) for i in range(5)],
        'dropoff_datetime': [
            base_time + timedelta(hours=0, minutes=30),  # Normal
            base_time + timedelta(hours=1, seconds=30),  # Too short (30s)
            base_time + timedelta(hours=7),     # Too long (5 hours from hour 2)
            base_time + timedelta(hours=3, minutes=30),  # Normal
            base_time + timedelta(hours=4, minutes=30),  # Normal
        ],
        'passenger_count': [1, 2, 15, 1, 3],  # 15 is outlier
        'trip_distance': [1.5, 2.5, 200.0, 1.0, 2.0],  # 200 is outlier
        'pickup_longitude': [-73.98, -73.97, -73.99, -73.96, -73.95],
        'pickup_latitude': [40.75, 40.76, 40.74, 40.77, 40.73],
        'dropoff_longitude': [-73.97, -73.96, -73.98, -73.95, -73.94],
        'dropoff_latitude': [40.76, 40.77, 40.75, 40.78, 40.74],
        'RatecodeID': ['1', '1', '1', '1', '1'],
        'store_and_fwd_flag': ['N', 'N', 'N', 'N', 'N'],
        'payment_type': ['1', '2', '1', '1', '1'],
        'fare_amount': [10.0, 15.0, 1000.0, 8.0, 12.0],  # 1000 is outlier
        'extra': [0.5, 0.5, 0.0, 0.5, 0.5],
        'mta_tax': [0.5, 0.5, 0.5, 0.5, 0.5],
        'tip_amount': [2.0, 0.0, 3.0, 1.5, 2.5],
        'tolls_amount': [0.0, 5.0, 0.0, 0.0, 0.0],
        'improvement_surcharge': [0.3, 0.3, 0.3, 0.3, 0.3],
        'total_amount': [13.3, 21.3, 1003.8, 10.8, 15.8],
    })


class TestTaxiDataPreprocessor:
    """Test cases for TaxiDataPreprocessor class."""
    
    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        preprocessor = TaxiDataPreprocessor()
        assert preprocessor.standardize_columns is False
        assert preprocessor.remove_outliers is True
        assert preprocessor.create_features is False
        assert preprocessor.target_column == "trip_duration"
    
    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        preprocessor = TaxiDataPreprocessor(
            standardize_columns=True,
            remove_outliers=False,
            create_features=True,
            target_column="duration",
        )
        assert preprocessor.standardize_columns is True
        assert preprocessor.remove_outliers is False
        assert preprocessor.create_features is True
        assert preprocessor.target_column == "duration"
    
    def test_fit(self, sample_taxi_data):
        """Test fitting the preprocessor."""
        preprocessor = TaxiDataPreprocessor()
        result = preprocessor.fit(sample_taxi_data)
        
        assert result is preprocessor  # Should return self
        assert preprocessor.vendor_mapping_ is not None
        assert preprocessor.payment_mapping_ is not None
        assert preprocessor.ratecode_mapping_ is not None
    
    def test_transform_creates_trip_duration(self, sample_taxi_data):
        """Test that transform creates trip_duration column."""
        preprocessor = TaxiDataPreprocessor(remove_outliers=False)
        preprocessor.fit(sample_taxi_data)
        result = preprocessor.transform(sample_taxi_data)
        
        assert 'trip_duration' in result.columns
        assert all(result['trip_duration'] == 1800.0)  # 30 minutes = 1800 seconds
    
    def test_transform_removes_invalid_records(self):
        """Test that transform removes invalid records."""
        data = pd.DataFrame({
            'VendorID': ['1', '2', '1'],
            'pickup_datetime': [datetime(2010, 1, 1, 10, 0), datetime(2010, 1, 1, 11, 0), None],
            'dropoff_datetime': [datetime(2010, 1, 1, 10, 30), None, datetime(2010, 1, 1, 12, 30)],
            'passenger_count': [1, 0, 1],  # 0 is invalid
            'trip_distance': [1.5, -1.0, 2.0],  # -1.0 is invalid
            'fare_amount': [10.0, 15.0, 0.0],  # 0.0 is invalid
            'pickup_longitude': [-73.98, -73.97, -73.99],
            'pickup_latitude': [40.75, 40.76, 40.74],
            'dropoff_longitude': [-73.97, -73.96, -73.98],
            'dropoff_latitude': [40.76, 40.77, 40.75],
            'RatecodeID': ['1', '1', '1'],
            'store_and_fwd_flag': ['N', 'N', 'N'],
            'payment_type': ['1', '2', '1'],
        })
        
        preprocessor = TaxiDataPreprocessor(remove_outliers=False)
        preprocessor.fit(data)
        result = preprocessor.transform(data)
        
        # Should only keep the first record
        assert len(result) == 1
    
    def test_transform_removes_outliers(self, sample_taxi_data_with_outliers):
        """Test that transform removes outlier records when enabled."""
        preprocessor = TaxiDataPreprocessor(remove_outliers=True)
        preprocessor.fit(sample_taxi_data_with_outliers)
        result = preprocessor.transform(sample_taxi_data_with_outliers)
        
        # Should remove records with outliers
        assert len(result) < len(sample_taxi_data_with_outliers)
        
        # Check that remaining data is within expected ranges
        if len(result) > 0:
            assert all(result['trip_duration'] >= 60)
            assert all(result['trip_duration'] <= 10800)
            assert all(result['trip_distance'] >= 0.1)
            assert all(result['trip_distance'] <= 100)
    
    def test_transform_keeps_outliers_when_disabled(self, sample_taxi_data_with_outliers):
        """Test that transform keeps outliers when remove_outliers is False."""
        preprocessor = TaxiDataPreprocessor(remove_outliers=False)
        preprocessor.fit(sample_taxi_data_with_outliers)
        result = preprocessor.transform(sample_taxi_data_with_outliers)
        
        # Some records might still be removed due to invalid values
        # but not due to outlier filtering
        assert 'trip_duration' in result.columns
    
    def test_transform_creates_features(self, sample_taxi_data):
        """Test that transform creates engineered features when enabled."""
        preprocessor = TaxiDataPreprocessor(
            remove_outliers=False,
            create_features=True
        )
        preprocessor.fit(sample_taxi_data)
        result = preprocessor.transform(sample_taxi_data)
        
        # Check temporal features
        assert 'pickup_hour' in result.columns
        assert 'pickup_day' in result.columns
        assert 'pickup_weekday' in result.columns
        assert 'pickup_month' in result.columns
        assert 'pickup_year' in result.columns
        assert 'is_weekend' in result.columns
        assert 'is_morning_rush' in result.columns
        assert 'is_evening_rush' in result.columns
        assert 'is_night' in result.columns
        
        # Check distance features
        assert 'haversine_distance' in result.columns
        assert 'manhattan_distance' in result.columns
        assert 'bearing' in result.columns
        
        # Check categorical features
        assert 'is_airport_trip' in result.columns
    
    def test_transform_temporal_features(self, sample_taxi_data):
        """Test temporal feature engineering."""
        preprocessor = TaxiDataPreprocessor(
            remove_outliers=False,
            create_features=True
        )
        preprocessor.fit(sample_taxi_data)
        result = preprocessor.transform(sample_taxi_data)
        
        # First record is at 10:00 AM on 2010-01-01 (Friday)
        assert result.iloc[0]['pickup_hour'] == 10
        assert result.iloc[0]['pickup_day'] == 1
        assert result.iloc[0]['pickup_month'] == 1
        assert result.iloc[0]['pickup_year'] == 2010
        assert result.iloc[0]['pickup_weekday'] == 4  # Friday
        assert result.iloc[0]['is_weekend'] == 0
    
    def test_haversine_distance_calculation(self):
        """Test Haversine distance calculation."""
        lat1 = np.array([40.75])
        lon1 = np.array([-73.98])
        lat2 = np.array([40.76])
        lon2 = np.array([-73.97])
        
        distance = TaxiDataPreprocessor._calculate_haversine(lat1, lon1, lat2, lon2)
        
        # Distance should be positive and reasonable for NYC
        assert distance[0] > 0
        assert distance[0] < 10  # Should be less than 10 miles
    
    def test_bearing_calculation(self):
        """Test bearing calculation."""
        lat1 = np.array([40.75])
        lon1 = np.array([-73.98])
        lat2 = np.array([40.76])
        lon2 = np.array([-73.97])
        
        bearing = TaxiDataPreprocessor._calculate_bearing(lat1, lon1, lat2, lon2)
        
        # Bearing should be between 0 and 360 degrees
        assert 0 <= bearing[0] < 360
    
    def test_handle_missing_values(self, sample_taxi_data):
        """Test handling of missing values."""
        data = sample_taxi_data.copy()
        data.loc[0, 'passenger_count'] = None
        data.loc[1, 'extra'] = None
        
        preprocessor = TaxiDataPreprocessor(remove_outliers=False)
        preprocessor.fit(data)
        result = preprocessor.transform(data)
        
        # passenger_count should be filled with 1
        assert result.iloc[0]['passenger_count'] == 1
        # extra should be filled with 0
        assert result.iloc[1]['extra'] == 0
    
    def test_convert_data_types(self, sample_taxi_data):
        """Test data type conversion."""
        preprocessor = TaxiDataPreprocessor(remove_outliers=False)
        preprocessor.fit(sample_taxi_data)
        result = preprocessor.transform(sample_taxi_data)
        
        # Numeric columns should be numeric
        assert pd.api.types.is_numeric_dtype(result['passenger_count'])
        assert pd.api.types.is_numeric_dtype(result['trip_distance'])
        assert pd.api.types.is_numeric_dtype(result['fare_amount'])
        
        # store_and_fwd_flag should be 0 or 1
        assert all(result['store_and_fwd_flag'].isin([0, 1]))


class TestPreprocessTaxiData:
    """Test cases for preprocess_taxi_data convenience function."""
    
    def test_preprocess_taxi_data_basic(self, sample_taxi_data):
        """Test basic preprocessing."""
        result, preprocessor = preprocess_taxi_data(
            sample_taxi_data,
            remove_outliers=False,
            create_features=False
        )
        
        assert isinstance(result, pd.DataFrame)
        assert isinstance(preprocessor, TaxiDataPreprocessor)
        assert 'trip_duration' in result.columns
    
    def test_preprocess_taxi_data_with_features(self, sample_taxi_data):
        """Test preprocessing with feature creation."""
        result, preprocessor = preprocess_taxi_data(
            sample_taxi_data,
            remove_outliers=False,
            create_features=True
        )
        
        assert 'pickup_hour' in result.columns
        assert 'haversine_distance' in result.columns
    
    def test_preprocess_taxi_data_fit_preprocessor(self, sample_taxi_data):
        """Test preprocessing with fit_preprocessor=True."""
        result, preprocessor = preprocess_taxi_data(
            sample_taxi_data,
            remove_outliers=False,
            create_features=False,
            fit_preprocessor=True
        )
        
        assert preprocessor.vendor_mapping_ is not None
    
    def test_preprocess_taxi_data_custom_target(self, sample_taxi_data):
        """Test preprocessing with custom target column name."""
        result, preprocessor = preprocess_taxi_data(
            sample_taxi_data,
            remove_outliers=False,
            create_features=False,
            target_column='duration'
        )
        
        assert 'duration' in result.columns


class TestGetFeatureColumns:
    """Test cases for get_feature_columns function."""
    
    def test_get_feature_columns_without_target(self):
        """Test getting feature columns without target."""
        columns = get_feature_columns(include_target=False)
        
        assert isinstance(columns, list)
        assert len(columns) > 0
        assert 'trip_duration' not in columns
        assert 'pickup_hour' in columns
        assert 'passenger_count' in columns
    
    def test_get_feature_columns_with_target(self):
        """Test getting feature columns with target."""
        columns = get_feature_columns(include_target=True)
        
        assert isinstance(columns, list)
        assert 'trip_duration' in columns
