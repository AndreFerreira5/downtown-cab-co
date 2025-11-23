import os
import logging
import requests

logger = logging.getLogger(__name__)


class DataDownloader:
    def __init__(
        self,
        base_url="https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata",
        dest_folder="training/",
        years_to_download=["2011", "2012"],
    ):
        self.base_url = base_url
        self.dest_folder = dest_folder
        self.years_to_download = years_to_download

        os.makedirs(self.dest_folder, exist_ok=True)

    def download(self):
        for year in self.years_to_download:
            year_str = str(year)
            downloaded_files = []

            for month in range(1, 13):
                month_str = f"{month:02d}"
                url = f"{self.base_url}_{year_str}-{month_str}.parquet"
                filename = f"yellow_tripdata_{year_str}-{month_str}.parquet"
                filepath = os.path.join(self.dest_folder, filename)

                try:
                    response = requests.get(url, stream=True, timeout=(10, 30))
                    response.raise_for_status()
                    with open(filepath, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    logger.info(f"Downloaded {url}")
                    downloaded_files.append(filepath)
                except requests.exceptions.RequestException as e:
                    logger.error(f"Failed to download {url}: {e}")

        logger.info("finished downloading!")
