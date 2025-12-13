import os
import logging
import requests
import pandas as pd

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

    # Download the month data
    file_path = download_validation_month(month_num)
    if file_path is None:
        return None

    try:
        df = pd.read_parquet(file_path)

        # Filter to same day of month
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
        day_of_month = current_date.day
        matched = df[df['tpep_pickup_datetime'].dt.day == day_of_month].copy()

        # If no exact day match, use random sample
        if len(matched) == 0:
            matched = df.sample(n=min(1000, len(df))).copy()
            logger.info(f"Using {len(matched)} samples from month {month_num}")
        else:
            logger.info(f"Found {len(matched)} trips for day {day_of_month} in month {month_num}")

        return matched

    except Exception as e:
        logger.error(f"Error loading validation data: {e}")
        return None