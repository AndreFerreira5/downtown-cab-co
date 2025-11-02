#!/bin/bash

mlflow server \
  --host 0.0.0.0 \
  --port 3030 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root /mlflow/mlruns \
  --serve-artifacts \
  --allowed-hosts '*'