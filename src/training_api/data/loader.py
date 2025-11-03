import pandas as pd
from pathlib import Path
from typing import Optional
import glob
import os
import requests
import subprocess
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
        if download_dataset: self.download_dataset()
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

    def download_dataset(self):
        dest_folder = "training/"
        training_years = ["2010"]
        base_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata"
        os.makedirs(dest_folder, exist_ok=True)

        for year in training_years:
            year_str = str(year)
            downloaded_files = []

            for month in range(1, 13):
                month_str = f"{month:02d}"
                url = f"{base_url}_{year_str}-{month_str}.parquet"
                filename = f"yellow_tripdata_{year_str}-{month_str}.parquet"
                filepath = os.path.join(dest_folder, filename)

                try:
                    response = requests.get(url, stream=True, timeout=(10, 30))
                    response.raise_for_status()
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"Downloaded {url}")
                    downloaded_files.append(filepath)
                except requests.exceptions.RequestException as e:
                    print(f"Failed to download {url}: {e}")

                break # TODO remove this
        logger.info("finished downloading!")
