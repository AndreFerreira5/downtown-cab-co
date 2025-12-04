import pandas as pd
from pathlib import Path
from typing import Optional
import glob
from .downloader import DataDownloader
import logging
import gc
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


# TODO integrate with DVC and pull dataset files as needed?
# TODO calculate the dataset batch size according to free ram (having a maximum cap)
class DataLoader:
    """Load parquet files one at a time on demand."""

    def __init__(self, data_dir: str, batch_size: int = 50_000, download_dataset=False,
                 years_to_download=["2011", "2012"], verbose: bool = False):
        if download_dataset:
            DataDownloader(years_to_download=years_to_download, dest_folder=data_dir).download()

        self.data_dir = Path(data_dir)
        # Note: Parquet reads in 'Row Groups'. We target approx batch_size but exact size depends on the file structure.
        self.target_batch_size = batch_size
        self.verbose = verbose

        self.parquet_files = sorted(glob.glob(str(self.data_dir / "*.parquet")))
        if not self.parquet_files:
            raise ValueError(f"No parquet files found in {data_dir}")

        # State trackers
        self._current_file_idx = 0
        self._parquet_file_engine = None  # The Pyarrow reader engine
        self._batch_iterator = None  # The iterator over the current file

    def load_next_batch(self) -> Optional[pd.DataFrame]:
        """
        Streams the next batch directly from disk.
        """
        # 1. Open file if needed (Lazy Load)
        if self._batch_iterator is None:
            if self._current_file_idx >= len(self.parquet_files):
                return None  # Done with all files

            file_path = self.parquet_files[self._current_file_idx]
            if self.verbose: logger.info(
                f"Streaming file {self._current_file_idx + 1}/{len(self.parquet_files)}: {file_path}")

            # Open the file efficiently using PyArrow (Zero RAM load initially)
            self._parquet_file_engine = pq.ParquetFile(file_path)

            # Create an iterator that yields chunks
            self._batch_iterator = self._parquet_file_engine.iter_batches(batch_size=self.target_batch_size)

        # 2. Try to get the next chunk
        try:
            record_batch = next(self._batch_iterator)
            # Convert PyArrow batch to Pandas (Fast)
            return record_batch.to_pandas()

        except StopIteration:
            # Current file is finished. Move to next.
            self._batch_iterator = None
            self._parquet_file_engine = None
            self._current_file_idx += 1

            # Recurse to load from the new file immediately
            return self.load_next_batch()

    def reset(self) -> None:
        self._current_file_idx = 0
        self._batch_iterator = None
        self._parquet_file_engine = None
        gc.collect()

    def get_file_count(self) -> int:
        return len(self.parquet_files)
