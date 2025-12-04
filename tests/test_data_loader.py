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
    # Create 3 files with 1000 rows each
    for i in range(3):
        df = pd.DataFrame({
            'vendor_id': np.random.randint(1, 3, 1000),
            'tpep_pickup_datetime': pd.date_range('2011-01-01', periods=1000, freq='h'),
            'passenger_count': np.random.randint(1, 6, 1000),
            'trip_distance': np.random.uniform(0.5, 20, 1000),
            'PULocationID': np.random.randint(1, 265, 1000),
            'DOLocationID': np.random.randint(1, 265, 1000),
        })
        file_path = tmp_path / f"test_data_{i}.parquet"
        # Write with a small row_group_size to allow testing small batch streaming
        df.to_parquet(file_path, row_group_size=100, engine='pyarrow')
        files.append(file_path)

    return tmp_path, files


class TestDataLoader:
    """Test suite for DataLoader class."""

    def test_initialization(self, sample_parquet_files):
        """Test DataLoader initializes correctly."""
        data_dir, _ = sample_parquet_files
        loader = DataLoader(str(data_dir), batch_size=100)

        assert loader.target_batch_size == 100
        assert loader.get_file_count() == 3
        assert loader._current_file_idx == 0
        assert loader._parquet_file_engine is None

    def test_load_single_batch(self, sample_parquet_files):
        """Test loading a single batch."""
        data_dir, _ = sample_parquet_files
        loader = DataLoader(str(data_dir), batch_size=100)

        batch = loader.load_next_batch()

        assert batch is not None
        # PyArrow iter_batches is an upper limit, not strict size,
        # but with our synthetic data row_groups=100, it should match.
        assert len(batch) <= 100
        assert isinstance(batch, pd.DataFrame)
        assert 'vendor_id' in batch.columns

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

        # Total rows check (3 files * 1000 rows = 3000 rows)
        total_rows = sum(len(b) for b in batches)
        assert total_rows == 3000

        # Verify we actually got multiple batches
        assert len(batches) > 1

    def test_batch_integrity(self, sample_parquet_files):
        """Test that data loaded matches the source data exactly."""
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

        # Load original files manually to compare
        files = sorted(list(data_dir.glob("*.parquet")))
        original_dfs = [pd.read_parquet(f) for f in files]
        original_combined = pd.concat(original_dfs, ignore_index=True)

        # Sort both to ensure order doesn't affect equality check (though loader should preserve order)
        # Note: We reset index to ignore index differences
        pd.testing.assert_frame_equal(combined.reset_index(drop=True), original_combined.reset_index(drop=True))

    def test_reset_functionality(self, sample_parquet_files):
        """Test reset returns loader to initial state."""
        data_dir, _ = sample_parquet_files
        loader = DataLoader(str(data_dir), batch_size=100)

        # Load some batches
        loader.load_next_batch()
        loader.load_next_batch()

        # Ensure we advanced
        # (Internal implementation check, might vary based on row group size)
        assert loader._batch_iterator is not None

        # Reset
        loader.reset()

        # Verify internal state reset
        assert loader._current_file_idx == 0
        assert loader._batch_iterator is None
        assert loader._parquet_file_engine is None

    def test_file_transition(self, sample_parquet_files):
        """Test that loader moves to the next file correctly."""
        data_dir, _ = sample_parquet_files
        # Batch size 1000 (size of one file)
        loader = DataLoader(str(data_dir), batch_size=1000)

        # Load File 0
        batch1 = loader.load_next_batch()
        assert batch1 is not None
        # Should be reading file 0
        assert loader._current_file_idx == 0

        # Exhaust File 0 (if batch1 didn't take it all, which depends on row groups)
        # We loop until file index increments
        while loader._current_file_idx == 0:
            batch = loader.load_next_batch()
            if batch is None: break  # Should not happen yet

        # Now we should be on file 1
        assert loader._current_file_idx == 1

        # Ensure we still get data
        batch_new_file = loader.load_next_batch()
        assert batch_new_file is not None
        assert len(batch_new_file) > 0

    def test_empty_directory_raises_error(self, tmp_path):
        """Test that empty directory raises ValueError."""
        with pytest.raises(ValueError, match="No parquet files found"):
            DataLoader(str(tmp_path), batch_size=100)

    def test_file_count(self, sample_parquet_files):
        """Test get_file_count returns correct number."""
        data_dir, _ = sample_parquet_files
        loader = DataLoader(str(data_dir), batch_size=100)

        assert loader.get_file_count() == 3

    def test_consecutive_resets(self, sample_parquet_files):
        """Test multiple consecutive resets."""
        data_dir, _ = sample_parquet_files
        loader = DataLoader(str(data_dir), batch_size=100)

        first_batch = loader.load_next_batch()
        loader.reset()
        second_batch = loader.load_next_batch()
        loader.reset()
        third_batch = loader.load_next_batch()

        # All should be the same (first batch of the dataset)
        pd.testing.assert_frame_equal(first_batch, second_batch)
        pd.testing.assert_frame_equal(first_batch, third_batch)

    @pytest.mark.skipif(not hasattr(psutil, 'Process'), reason="psutil not available")
    def test_memory_usage_stays_bounded(self, tmp_path):
        """Test that memory doesn't grow unbounded when processing multiple files."""
        # Create larger test files to force memory pressure if leaks exist
        for i in range(5):
            df = pd.DataFrame({
                'data': np.random.randn(20000)  # 20k rows
            })
            # Force small row groups so we have many batches
            df.to_parquet(tmp_path / f"large_{i}.parquet", row_group_size=2000)

        loader = DataLoader(str(tmp_path), batch_size=5000)
        process = psutil.Process(os.getpid())

        memory_readings = []

        # Baseline memory
        gc.collect()
        start_mem = process.memory_info().rss / 1024 / 1024

        while True:
            batch = loader.load_next_batch()
            if batch is None:
                break

            # Force GC to simulate real loop cleanup
            del batch
            gc.collect()

            memory_mb = process.memory_info().rss / 1024 / 1024
            memory_readings.append(memory_mb)

        # Check for aggressive growth.
        # Streaming loader should stay relatively flat (variance allowed).
        # We allow 2x growth to account for Python overhead/fragmentation,
        # but a leak would be 10x+.
        max_mem = max(memory_readings) if memory_readings else start_mem

        # This assertion is heuristic but catches massive leaks
        assert max_mem < start_mem + 500, f"Memory grew too much: Start {start_mem}MB, Max {max_mem}MB"