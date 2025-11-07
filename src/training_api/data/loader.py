import pandas as pd
from pathlib import Path
from typing import Optional
import glob
from .downloader import DataDownloader
import logging

logger = logging.getLogger(__name__)


# TODO integrate with DVC and pull dataset files as needed?
# TODO calculate the dataset batch size according to free ram (having a maximum cap)
class DataLoader:
    """Load parquet files in batches on demand."""

    def __init__(self, data_dir: str, batch_size: int = 50_000, download_dataset=False):
        """
        Initialize the data loader.

        Args:
            data_dir: Directory containing parquet files
            batch_size: Number of rows to load per batch
        """
        if download_dataset:
            DataDownloader().download()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.parquet_files = sorted(glob.glob(str(self.data_dir / "*.parquet")))

        if not self.parquet_files:
            raise ValueError(f"No parquet files found in {data_dir}")

        # Track current position
        self._current_file_idx = 0
        self._current_row_idx = 0
        self._current_file_data: Optional[pd.DataFrame] = None

    def load_next_batch(self) -> Optional[pd.DataFrame]:
        """
        Load the next batch of data.

        Returns:
            DataFrame with the next batch, or None if all data has been read
        """
        # Load new file if needed
        if self._current_file_data is None:
            if self._current_file_idx >= len(self.parquet_files):
                return None

            file_path = self.parquet_files[self._current_file_idx]
            self._current_file_data = pd.read_parquet(file_path)
            self._current_row_idx = 0

        # Extract batch
        start_idx = self._current_row_idx
        end_idx = min(start_idx + self.batch_size, len(self._current_file_data))
        batch = self._current_file_data.iloc[start_idx:end_idx].copy()

        # Update position
        self._current_row_idx = end_idx

        # Move to next file if current is exhausted
        if self._current_row_idx >= len(self._current_file_data):
            self._current_file_idx += 1
            self._current_file_data = None

        return batch

    def reset(self) -> None:
        """Reset the loader to start from the beginning."""
        self._current_file_idx = 0
        self._current_row_idx = 0
        self._current_file_data = None

    def get_file_count(self) -> int:
        """Return the number of parquet files."""
        return len(self.parquet_files)
