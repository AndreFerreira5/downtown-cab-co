#!/bin/bash

mlflow server \
  --host 0.0.0.0 \
  --port 3030 \
  --backend-store-uri sqlite:////mlflow/mlflow.db \
  --default-artifact-root mlflow-artifacts:/ \
  --artifacts-destination /mlflow/mlruns \
  --allowed-hosts "*" \
  --serve-artifacts