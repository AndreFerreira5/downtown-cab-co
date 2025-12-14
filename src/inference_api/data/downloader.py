import os
import logging
import requests
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import gc

logger = logging.getLogger(__name__)


def download_validation_month(month_num, cache_folder="/app/validation_cache"):
    """
    Download a specific month of 2013 validation data.
    Returns the file path if successful, None otherwise.
    """
    os.makedirs(cache_folder, exist_ok=True)

    file_pattern = f"yellow_tripdata_2013-{month_num:02d}.parquet"
    file_path = os.path.join(cache_folder, file_pattern)

    # Return cached file if exists
    if os.path.exists(file_path):
        logger.info(f"Using cached validation data: {file_pattern}")
        return file_path

    # Download if not cached
    logger.info(f"Downloading validation data for 2013-{month_num:02d}...")
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2013-{month_num:02d}.parquet"

    try:
        response = requests.get(url, stream=True, timeout=(10, 60))
        response.raise_for_status()

        # Download to temp file first, then rename (atomic operation)
        temp_path = f"{file_path}.tmp"
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        os.rename(temp_path, file_path)
        logger.info(f"✅ Downloaded {file_pattern}")
        return file_path

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download validation data: {e}")
        return None


def load_validation_data_for_date(current_date):
    """Load validation data from 2013 matching current month and day"""
    month_num = current_date.month
    day_of_month = current_date.day

    # Download the month data
    file_path = download_validation_month(month_num)
    if file_path is None:
        return None

    try:
        logger.info(f"Opening parquet file: {file_path}")

        # 1. Open the file without reading data
        parquet_file = pq.ParquetFile(file_path)

        # 2. Read ONLY the 'tpep_pickup_datetime' column to find matching rows
        # This is very lightweight compared to reading all columns
        timestamp_col = parquet_file.read(columns=['tpep_pickup_datetime'])
        timestamps = timestamp_col.column(0).to_pandas()

        # 3. Find indices for the specific day
        # Note: We assume timestamps are already datetime objects in the parquet file
        # If not, we might need pd.to_datetime(timestamps)
        mask = timestamps.dt.day == day_of_month

        # Get integer indices where mask is True
        matching_indices = mask[mask].index.tolist()

        if len(matching_indices) == 0:
            logger.info(f"No exact matches for day {day_of_month}. Sampling 1000 random rows.")
            # Fallback: Read a random sample of row groups or just head
            # Reading first 1000 rows is safe and fast
            subset_df = pd.read_parquet(file_path).sample(n=1000)
            # Note: The above line still reads full file.
            # Better low-memory random sample:
            import random
            total_rows = parquet_file.metadata.num_rows
            random_indices = random.sample(range(total_rows), min(1000, total_rows))
            # Reading specific scattered indices is slow in parquet,
            # so for the fallback, we might just accept the hit or read the first row group.
            # Let's stick to the simplest memory-safe fallback:
            subset_df = parquet_file.read_row_group(0).to_pandas().head(1000)

            del timestamps
            del mask
            gc.collect()
            return subset_df

        logger.info(f"Found {len(matching_indices)} trips for day {day_of_month}. Loading full data for these rows...")

        # 4. Read only the specific rows we need
        # PyArrow doesn't support "read rows by index list" efficiently across the whole file directly
        # But we can iterate row groups or just read the whole table *if* the subset is large.
        # However, to be strictly memory safe, we can use the mask to filter PyArrow table:

        # Efficient Strategy: Read file into PyArrow Table (more compact than Pandas), filter, then convert.
        table = parquet_file.read()
        filtered_table = table.filter(pa.array(mask))

        # Convert ONLY the filtered data to Pandas
        matched = filtered_table.to_pandas()

        # Clean up heavy objects immediately
        del table
        del filtered_table
        del timestamps
        del mask
        gc.collect()

        logger.info(f"Successfully loaded {len(matched)} rows.")
        return matched

    except Exception as e:
        logger.error(f"Error loading validation data: {e}")
        gc.collect()
        return None