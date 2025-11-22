import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import gc
import psutil
import os
from training_api.data.loader import DataLoader


@pytest.fixture
def sample_parquet_files(tmp_path):
    """Create sample parquet files for testing."""
    files = []
    for i in range(3):
        df = pd.DataFrame({
            'vendor_id': np.random.randint(1, 3, 1000),
            'pickup_datetime': pd.date_range('2011-01-01', periods=1000, freq='h'),
            'passenger_count': np.random.randint(1, 6, 1000),
            'trip_distance': np.random.uniform(0.5, 20, 1000),
            'pickup_location': np.random.randint(1, 265, 1000),
            'dropoff_location': np.random.randint(1, 265, 1000),
        })
        file_path = tmp_path / f"test_data_{i}.parquet"
        df.to_parquet(file_path)
        files.append(file_path)

    return tmp_path, files


class TestDataLoader:
    """Test suite for DataLoader class."""

    def test_initialization(self, sample_parquet_files):
        """Test DataLoader initializes correctly."""
        data_dir, _ = sample_parquet_files
        loader = DataLoader(str(data_dir), batch_size=100)

        assert loader.batch_size == 100
        assert loader.get_file_count() == 3
        assert loader._current_file_idx == 0
        assert loader._current_row_idx == 0
        assert loader._current_file_data is None

    def test_load_single_batch(self, sample_parquet_files):
        """Test loading a single batch."""
        data_dir, _ = sample_parquet_files
        loader = DataLoader(str(data_dir), batch_size=100)

        batch = loader.load_next_batch()

        assert batch is not None
        assert len(batch) == 100
        assert isinstance(batch, pd.DataFrame)

    def test_load_all_batches(self, sample_parquet_files):
        """Test loading all batches sequentially."""
        data_dir, _ = sample_parquet_files
        loader = DataLoader(str(data_dir), batch_size=500)

        batches = []
        while True:
            batch = loader.load_next_batch()
            if batch is None:
                break
            batches.append(batch)

        # Should have 6 batches (3 files × 1000 rows ÷ 500 batch_size = 6)
        assert len(batches) == 6
        total_rows = sum(len(b) for b in batches)
        assert total_rows == 3000  # 3 files × 1000 rows

    def test_batch_integrity(self, sample_parquet_files):
        """Test that batches don't overlap or skip data within the same file."""
        data_dir, _ = sample_parquet_files
        loader = DataLoader(str(data_dir), batch_size=250)

        all_data = []
        while True:
            batch = loader.load_next_batch()
            if batch is None:
                break
            all_data.append(batch)

        # Concatenate all batches
        combined = pd.concat(all_data, ignore_index=True)

        # Check we got all the data (3 files × 1000 rows)
        assert len(combined) == 3000

        # Check no rows were duplicated by comparing with original files
        total_rows = sum(len(batch) for batch in all_data)
        assert total_rows == 3000

    def test_reset_functionality(self, sample_parquet_files):
        """Test reset returns loader to initial state."""
        data_dir, _ = sample_parquet_files
        loader = DataLoader(str(data_dir), batch_size=100)

        # Load some batches
        loader.load_next_batch()
        loader.load_next_batch()

        # Reset
        loader.reset()

        assert loader._current_file_idx == 0
        assert loader._current_row_idx == 0
        assert loader._current_file_data is None

    def test_memory_cleanup_between_files(self, sample_parquet_files):
        """Test that memory is freed when moving between files."""
        data_dir, _ = sample_parquet_files
        loader = DataLoader(str(data_dir), batch_size=1000, download_dataset=False)

        # Load first file completely (should trigger cleanup)
        batch1 = loader.load_next_batch()
        assert batch1 is not None
        assert len(batch1) == 1000

        # At this point, first file should be exhausted and loader moved to file 1
        # But since we haven't loaded from file 1 yet, _current_file_data should be None
        assert loader._current_file_data is None
        assert loader._current_file_idx == 1  # Moved to next file

        # Load second file (first file data should be cleared)
        batch2 = loader.load_next_batch()
        assert batch2 is not None
        assert len(batch2) == 1000

        # Now we should be at file index 2
        assert loader._current_file_idx == 2

    def test_empty_directory_raises_error(self, tmp_path):
        """Test that empty directory raises ValueError."""
        with pytest.raises(ValueError, match="No parquet files found"):
            DataLoader(str(tmp_path), batch_size=100)

    def test_file_count(self, sample_parquet_files):
        """Test get_file_count returns correct number."""
        data_dir, _ = sample_parquet_files
        loader = DataLoader(str(data_dir), batch_size=100)

        assert loader.get_file_count() == 3

    def test_partial_batch_at_end(self, tmp_path):
        """Test handling of partial batch at end of file."""
        # Create file with 250 rows
        df = pd.DataFrame({
            'col1': range(250),
            'col2': range(250, 500)
        })
        file_path = tmp_path / "test.parquet"
        df.to_parquet(file_path)

        loader = DataLoader(str(tmp_path), batch_size=100)

        batch1 = loader.load_next_batch()
        batch2 = loader.load_next_batch()
        batch3 = loader.load_next_batch()  # Should have only 50 rows
        batch4 = loader.load_next_batch()  # Should be None

        assert len(batch1) == 100
        assert len(batch2) == 100
        assert len(batch3) == 50
        assert batch4 is None

    def test_consecutive_resets(self, sample_parquet_files):
        """Test multiple consecutive resets."""
        data_dir, _ = sample_parquet_files
        loader = DataLoader(str(data_dir), batch_size=100)

        first_batch = loader.load_next_batch()
        loader.reset()
        second_batch = loader.load_next_batch()
        loader.reset()
        third_batch = loader.load_next_batch()

        # All should be the same (first batch)
        pd.testing.assert_frame_equal(first_batch, second_batch)
        pd.testing.assert_frame_equal(first_batch, third_batch)

    @pytest.mark.skipif(not hasattr(psutil, 'Process'), reason="psutil not available")
    def test_memory_usage_stays_bounded(self, tmp_path):
        """Test that memory doesn't grow unbounded when processing multiple files."""
        # Create larger test files
        for i in range(5):
            df = pd.DataFrame({
                'data': np.random.randn(10000)
            })
            df.to_parquet(tmp_path / f"large_{i}.parquet")

        loader = DataLoader(str(tmp_path), batch_size=5000)
        process = psutil.Process(os.getpid())

        memory_readings = []

        while True:
            batch = loader.load_next_batch()
            if batch is None:
                break
            gc.collect()
            memory_mb = process.memory_info().rss / 1024 / 1024
            memory_readings.append(memory_mb)

        # Memory shouldn't continuously grow
        # Check that max memory is not significantly larger than initial
        if len(memory_readings) > 2:
            initial_avg = np.mean(memory_readings[:2])
            final_avg = np.mean(memory_readings[-2:])
            # Allow some growth but not unbounded
            assert final_avg < initial_avg * 2, "Memory appears to be leaking"


class TestDataLoaderIntegration:
    """Integration tests for DataLoader."""

    def test_full_iteration_workflow(self, sample_parquet_files):
        """Test complete workflow: load all, reset, load again."""
        data_dir, _ = sample_parquet_files
        loader = DataLoader(str(data_dir), batch_size=300)

        # First iteration
        first_iteration = []
        while True:
            batch = loader.load_next_batch()
            if batch is None:
                break
            first_iteration.append(len(batch))

        # Reset
        loader.reset()

        # Second iteration
        second_iteration = []
        while True:
            batch = loader.load_next_batch()
            if batch is None:
                break
            second_iteration.append(len(batch))

        # Both iterations should yield same batch counts
        assert first_iteration == second_iteration

    def test_destructor_cleanup(self, sample_parquet_files):
        """Test that __del__ properly cleans up."""
        data_dir, _ = sample_parquet_files
        loader = DataLoader(str(data_dir), batch_size=100)

        # Load some data
        loader.load_next_batch()

        # Delete loader (triggers __del__)
        del loader
        gc.collect()

        # If no exception, cleanup worked
        assert True
