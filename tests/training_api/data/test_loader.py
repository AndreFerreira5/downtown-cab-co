"""Unit tests for data loader module."""

import pytest
import pandas as pd
import os
import tempfile
from pathlib import Path
from src.training_api.data.loader import DataLoader


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory with test parquet files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample data
        df1 = pd.DataFrame({
            'pickup_datetime': pd.date_range('2010-01-01', periods=100, freq='h'),
            'dropoff_datetime': pd.date_range('2010-01-01 00:30:00', periods=100, freq='h'),
            'trip_distance': [1.5] * 100,
            'fare_amount': [10.0] * 100,
        })
        df2 = pd.DataFrame({
            'pickup_datetime': pd.date_range('2010-01-05', periods=150, freq='h'),
            'dropoff_datetime': pd.date_range('2010-01-05 00:30:00', periods=150, freq='h'),
            'trip_distance': [2.0] * 150,
            'fare_amount': [15.0] * 150,
        })
        
        # Save to parquet files
        df1.to_parquet(os.path.join(tmpdir, 'data1.parquet'))
        df2.to_parquet(os.path.join(tmpdir, 'data2.parquet'))
        
        yield tmpdir


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    def test_init_with_valid_directory(self, temp_data_dir):
        """Test initialization with valid directory containing parquet files."""
        loader = DataLoader(temp_data_dir, batch_size=50, download_dataset=False)
        assert loader.batch_size == 50
        assert len(loader.parquet_files) == 2
        assert loader._current_file_idx == 0
        assert loader._current_row_idx == 0
    
    def test_init_with_empty_directory(self):
        """Test initialization with directory containing no parquet files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No parquet files found"):
                DataLoader(tmpdir, download_dataset=False)
    
    def test_load_next_batch_first_batch(self, temp_data_dir):
        """Test loading the first batch of data."""
        loader = DataLoader(temp_data_dir, batch_size=50, download_dataset=False)
        batch = loader.load_next_batch()
        
        assert batch is not None
        assert len(batch) == 50
        assert 'pickup_datetime' in batch.columns
        assert 'trip_distance' in batch.columns
    
    def test_load_next_batch_multiple_batches(self, temp_data_dir):
        """Test loading multiple batches from the same file."""
        loader = DataLoader(temp_data_dir, batch_size=40, download_dataset=False)
        
        batch1 = loader.load_next_batch()
        assert len(batch1) == 40
        
        batch2 = loader.load_next_batch()
        assert len(batch2) == 40
        
        # Third batch should be smaller (remaining 20 from first file)
        batch3 = loader.load_next_batch()
        assert len(batch3) == 20
    
    def test_load_next_batch_across_files(self, temp_data_dir):
        """Test loading batches that span across multiple files."""
        loader = DataLoader(temp_data_dir, batch_size=120, download_dataset=False)
        
        # First batch from first file (only 100 rows)
        batch1 = loader.load_next_batch()
        assert len(batch1) == 100
        
        # Second batch from second file
        batch2 = loader.load_next_batch()
        assert len(batch2) == 120
    
    def test_load_next_batch_until_exhausted(self, temp_data_dir):
        """Test loading batches until all data is exhausted."""
        loader = DataLoader(temp_data_dir, batch_size=100, download_dataset=False)
        
        batches = []
        while (batch := loader.load_next_batch()) is not None:
            batches.append(batch)
        
        # Total records: 100 + 150 = 250
        # With batch_size=100: 3 batches (100, 100, 50)
        assert len(batches) == 3
        assert len(batches[0]) == 100
        assert len(batches[1]) == 100
        assert len(batches[2]) == 50
    
    def test_load_next_batch_returns_none_when_exhausted(self, temp_data_dir):
        """Test that load_next_batch returns None after all data is consumed."""
        loader = DataLoader(temp_data_dir, batch_size=1000, download_dataset=False)
        
        # Load all batches
        while loader.load_next_batch() is not None:
            pass
        
        # Next call should return None
        assert loader.load_next_batch() is None
    
    def test_reset(self, temp_data_dir):
        """Test resetting the loader to start from the beginning."""
        loader = DataLoader(temp_data_dir, batch_size=50, download_dataset=False)
        
        # Load some batches
        loader.load_next_batch()
        loader.load_next_batch()
        
        # Reset
        loader.reset()
        
        assert loader._current_file_idx == 0
        assert loader._current_row_idx == 0
        assert loader._current_file_data is None
    
    def test_reset_and_reload(self, temp_data_dir):
        """Test that reset allows reloading data from the beginning."""
        loader = DataLoader(temp_data_dir, batch_size=50, download_dataset=False)
        
        # Load first batch
        batch1 = loader.load_next_batch()
        first_value = batch1.iloc[0]['trip_distance']
        
        # Load more batches
        loader.load_next_batch()
        loader.load_next_batch()
        
        # Reset and load first batch again
        loader.reset()
        batch1_again = loader.load_next_batch()
        
        assert batch1_again.iloc[0]['trip_distance'] == first_value
        assert len(batch1_again) == len(batch1)
    
    def test_get_file_count(self, temp_data_dir):
        """Test getting the number of parquet files."""
        loader = DataLoader(temp_data_dir, download_dataset=False)
        assert loader.get_file_count() == 2
    
    def test_batch_size_parameter(self, temp_data_dir):
        """Test that batch_size parameter is correctly set."""
        loader = DataLoader(temp_data_dir, batch_size=25, download_dataset=False)
        batch = loader.load_next_batch()
        
        assert loader.batch_size == 25
        assert len(batch) == 25
    
    def test_data_integrity(self, temp_data_dir):
        """Test that loaded data maintains integrity."""
        loader = DataLoader(temp_data_dir, batch_size=50, download_dataset=False)
        batch = loader.load_next_batch()
        
        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(batch['pickup_datetime'])
        assert pd.api.types.is_datetime64_any_dtype(batch['dropoff_datetime'])
        assert pd.api.types.is_numeric_dtype(batch['trip_distance'])
        assert pd.api.types.is_numeric_dtype(batch['fare_amount'])
    
    def test_batch_is_copy(self, temp_data_dir):
        """Test that returned batch is a copy, not a view."""
        loader = DataLoader(temp_data_dir, batch_size=50, download_dataset=False)
        batch = loader.load_next_batch()
        
        # Modify the batch
        original_value = batch.iloc[0]['trip_distance']
        batch.iloc[0, batch.columns.get_loc('trip_distance')] = 999.0
        
        # Reset and reload
        loader.reset()
        batch_again = loader.load_next_batch()
        
        # Original data should be unchanged
        assert batch_again.iloc[0]['trip_distance'] == original_value
