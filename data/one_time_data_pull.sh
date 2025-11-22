#!/bin/bash

TRAINING_YEARS=("2011" "2012")
TESTING_YEARS=("2013")

download_year_data() {
  local year=$1
  local dest_folder=$2
  local base_url="https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata"

  for month in {01..12}; do
    url="${base_url}_${year}-${month}.parquet"
    wget "$url" -P "${dest_folder}" || echo "failed to download ${url}"
  done
}

for year in "${TRAINING_YEARS[@]}"; do
  download_year_data "${year}" "training/"
  dvc add "training/yellow_tripdata_${year}*"
done

#for year in "${TESTING_YEARS[@]}"; do
#  download_year_data "${year}" "testing/"
#  dvc add "testing/yellow_tripdata_${year}*"
#done

dvc push