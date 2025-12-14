# 🚕 Downtown Cab Co.  - NYC Taxi Trip Duration Prediction

[![CI/CD Pipeline](https://github.com/AndreFerreira5/downtown-cab-co/actions/workflows/1-continuous-integration.yml/badge.svg)](https://github.com/AndreFerreira5/downtown-cab-co/actions/workflows/1-continuous-integration.yml)
[![License:  MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end **MLOps pipeline** for predicting taxi trip durations in New York City.  This project demonstrates production-grade machine learning engineering practices including automated training, experiment tracking, containerized deployment, monitoring, and drift detection.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Components](#-components)
  - [Training Pipeline](#training-pipeline)
  - [Inference API](#inference-api)
  - [MLflow Server](#mlflow-server)
  - [Model Promotion](#model-promotion)
- [MLOps Pipeline](#-mlops-pipeline)
  - [Continuous Integration](#1-continuous-integration)
  - [Continuous Delivery](#2-continuous-delivery)
  - [Continuous Staging](#3-continuous-staging)
  - [Continuous Deployment](#4-continuous-deployment)
- [Model Architecture](#-model-architecture)
- [Data Pipeline](#-data-pipeline)
- [Monitoring & Drift Detection](#-monitoring--drift-detection)
- [DVC Integration](#-dvc-integration)
- [Infrastructure](#-infrastructure)
- [Getting Started](#-getting-started)
- [Technologies](#-technologies)
- [Team](#-team)
- [License](#-license)

---

## 🎯 Overview

This project implements a complete MLOps workflow to predict the duration of taxi trips in New York City using real data from the NYC Taxi and Limousine Commission (TLC). The system is designed to handle: 

- **Automated model training** with hyperparameter optimization
- **Experiment tracking and model versioning** via MLflow
- **Containerized deployment** using Docker
- **CI/CD automation** through GitHub Actions with self-hosted runners
- **Performance monitoring** and automated drift detection
- **Data versioning** with DVC and Google Drive

### Key Features

✅ Hybrid ML model (Ridge Regression + LightGBM) optimized for resource-constrained environments  
✅ Memory-efficient batch streaming for large datasets  
✅ Fully automated CI/CD pipeline with 4 sequential stages  
✅ Real-time inference API with `/predict` endpoint  
✅ Automated retraining triggered by performance drift  
✅ Data versioning and lineage tracking with DVC  

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GitHub Repository                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Source    │  │   GitHub    │  │    DVC      │  │   Docker Images     │ │
│  │    Code     │  │   Actions   │  │  Metadata   │  │   (GHCR)            │ │
│  └─────────────┘  └──────┬──────┘  └─────────────┘  └─────────────────────┘ │
└──────────────────────────┼──────────────────────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   VM1 (Train)   │ │  Google Drive   │ │  VM2 (Inference)│
│  ┌───────────┐  │ │  ┌───────────┐  │ │  ┌───────────┐  │
│  │  MLflow   │  │ │  │   DVC     │  │ │  │ Inference │  │
│  │  Server   │  │ │  │  Remote   │  │ │  │    API    │  │
│  └───────────┘  │ │  │  Storage  │  │ │  └───────────┘  │
│  ┌───────────┐  │ │  └───────────┘  │ │  ┌───────────┐  │
│  │ Training  │  │ └─────────────────┘ │  │Monitoring │  │
│  │  Script   │  │                     │  │  & Drift  │  │
│  └───────────┘  │                     │  └───────────┘  │
│  ┌───────────┐  │                     └─────────────────┘
│  │  GitHub   │  │
│  │  Runner   │  │
│  └───────────┘  │
└─────────────────┘
```

---

## 📁 Project Structure

```
downtown-cab-co/
├── .github/
│   └── workflows/
│       ├── 1-continuous-integration.yml    # Testing & building
│       ├── 2-continuous-delivery.yml       # Training & staging
│       ├── 3-continuous-staging.yml        # E2E testing & promotion
│       ├── 4-continuous-deployment.yml     # Production deployment
│       └── scheduled-validation.yml        # Drift detection trigger
├── src/
│   ├── inference_api/          # FastAPI inference service
│   │   ├── data/
│   │   │   └── downloader.py   # Validation data retrieval
│   │   ├── Dockerfile
│   │   └── main.py             # API endpoints
│   ├── mlflow-server/          # MLflow tracking server
│   │   └── Dockerfile
│   └── training_api/           # Model training service
│       ├── data/
│       │   ├── downloader.py   # Dataset downloader
│       │   ├── loader.py       # Batch streaming loader
│       │   └── processer.py    # Data preprocessing
│       ├── Dockerfile
│       ├── config.py           # Training configuration
│       ├── main.py             # Training entrypoint
│       ├── train.py            # Model training logic
│       └── test.py             # Model evaluation
├── model-promotion/
│   └── promote_model.py        # MLflow model promotion script
├── tests/                      # Unit and E2E tests
├── data/                       # Local data directory
├── *.compose.*. yml             # Docker Compose configurations
├── training. dvc                # DVC tracking for training data
├── testing.dvc                 # DVC tracking for test data
├── pyproject.toml              # Python project configuration
└── uv.lock                     # Dependency lock file
```

---

## 🔧 Components

### Training Pipeline

The training component (`src/training_api/`) handles the complete model training lifecycle:

| Module | Description |
|--------|-------------|
| `main.py` | Orchestrates hyperparameter tuning and main training |
| `train.py` | Implements the hybrid model architecture and training logic |
| `config.py` | Manages environment variables and MLflow configuration |
| `data/loader.py` | Memory-efficient batch streaming from Parquet files |
| `data/downloader.py` | Automated dataset acquisition with retry logic |
| `data/processer.py` | Data cleaning, validation, and feature engineering |

**Key Features:**
- **Lazy Loading**: Files are indexed but only loaded on demand
- **Batch Streaming**: Processes 50,000 rows at a time using PyArrow
- **Two-Phase Training**: Hyperparameter search on 1% data, full training on 10%

### Inference API

The inference service (`src/inference_api/`) provides real-time predictions:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Predict trip duration from raw input |
| `/reload` | POST | Reload model from MLflow registry |
| `/validate` | POST | Trigger validation and drift detection |
| `/health` | GET | Health check endpoint |

**Input Format:**
```json
{
  "tpep_pickup_datetime": "2013-01-15 08:30:00",
  "PULocationID": 161,
  "DOLocationID": 237,
  "trip_distance":  3.5,
  "passenger_count": 2
}
```

**Output Format:**
```json
{
  "predicted_duration_minutes": 12.5
}
```

### MLflow Server

The MLflow server (`src/mlflow-server/`) provides:
- **Experiment Tracking**: Logs parameters, metrics, and artifacts
- **Model Registry**: Manages model versions with staging/production aliases
- **Artifact Storage**: Persists trained models and metadata

### Model Promotion

The promotion script (`model-promotion/promote_model.py`) handles model lifecycle transitions: 
- Retrieves models by alias (e.g., `staging`)
- Promotes to target alias (e.g., `production`)
- Integrates with CI/CD for automated promotion after E2E tests

---

## 🔄 MLOps Pipeline

The pipeline consists of 4 sequential stages, each triggered upon successful completion of the previous: 

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   1.  Continuous  │───▶│   2. Continuous  │───▶│   3. Continuous  │───▶│   4. Continuous  │
│   Integration    │    │    Delivery      │    │     Staging      │    │   Deployment     │
└──────────────────┘    └──────────────────┘    └──────────────────┘    └──────────────────┘
       │                        │                       │                       │
       ▼                        ▼                       ▼                       ▼
   Unit Tests              Train Model             E2E Tests              Reload API
   Build Images            Tag:  staging          Promote Model           Production
   Push to GHCR           Log to MLflow         Tag: production          Ready
```

### 1. Continuous Integration

**File:** `.github/workflows/1-continuous-integration. yml`

- Runs unit tests for all Python components
- Detects changed files to conditionally build only affected packages
- Builds Docker images for: 
  - MLflow Server
  - Training API
  - Inference API
- Pushes images to GitHub Container Registry (GHCR)
- Runs on GitHub-hosted runners (no VM access needed)

### 2. Continuous Delivery

**File:** `.github/workflows/2-continuous-delivery. yml`

- Runs on **VM1** (Training VM) via self-hosted runner
- Pulls training data using DVC from Google Drive
- Executes the training container: 
  1. Hyperparameter grid search (1% sample)
  2. Main training run (10% sample)
- Logs all experiments to MLflow
- Tags the trained model with `staging` alias

### 3. Continuous Staging

**File:** `.github/workflows/3-continuous-staging.yml`

- Runs on **VM1** via self-hosted runner
- Deploys a temporary inference API container
- Executes end-to-end tests against the staged model
- On success:  promotes model to `production` alias
- Cleans up temporary containers

### 4. Continuous Deployment

**File:** `.github/workflows/4-continuous-deployment.yml`

- Runs on **VM2** (Inference VM) via self-hosted runner
- Calls the `/reload` endpoint on the production API
- The API fetches the latest `production` model from MLflow
- Zero-downtime deployment

---

## 🧠 Model Architecture

The system uses a **hybrid ensemble** approach optimized for resource-constrained environments: 

### Component 1: Trend Learner (Ridge Regression)

**Purpose:** Captures macro-level temporal patterns and seasonality

**Features:**
- Linear time trend (`date_int`)
- Seasonal indicators (`sin_time`, `cos_time`)

**Target:** Log-transformed average daily trip duration

### Component 2: Contextual Learner (LightGBM)

**Purpose:** Models micro-level trip complexity and route details

**Features:**
- Trip distance
- Pickup hour and day of week
- Location IDs (pickup/dropoff zones)
- Rush hour flags, holiday indicators

**Target:** Log-ratio between actual duration and trend prediction

### Inference Combination

```python
# Final prediction combines both models
trend_prediction = exp(trend_model.predict(X_trend))
ratio_multiplier = exp(booster_model.predict(X_contextual))
final_duration = trend_prediction * ratio_multiplier
```

### Performance Metrics

| Metric | Value |
|--------|-------|
| MAE | 157.15 seconds (~2.6 minutes) |
| RMSE | 257.18 seconds (~4.3 minutes) |
| R² | 0.73 |

---

## 📊 Data Pipeline

### Dataset

- **Source:** [NYC Taxi & Limousine Commission (TLC)](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- **Training Data:** Yellow Taxi trips from 2011-2012
- **Validation/Testing:** Yellow Taxi trips from 2013

### Data Loading

The `DataLoader` class implements memory-efficient batch streaming from disk

### Preprocessing Pipeline

The `TaxiDataPreprocessor` performs:

1. **Column Standardization:** Handles naming inconsistencies across years
2. **Outlier Removal:**
   - Distance:  0.1 - 100 miles
   - Duration: 1 minute - 3 hours
   - Speed: 0.5 - 65 mph
3. **Type Conversion:** Ensures numeric types for ML
4. **Default Value Imputation:** Passenger count = 1, Location IDs = 0
5. **Feature Engineering:**
   - Temporal:  hour, weekday, month
   - Contextual: rush hour flags, holidays, night trips

---

## 📈 Monitoring & Drift Detection

### Performance Monitoring

The inference API implements a sliding window RMSE tracker

### Automated Drift Detection

**Trigger:** Scheduled GitHub Action (`scheduled-validation.yml`)

**Process:**
1. Scheduled action calls `/validate` endpoint
2. API downloads corresponding 2013 data (matching month/day)
3. Generates predictions and compares to ground truth
4. Calculates moving average RMSE
5. If `RMSE > threshold`: triggers retraining via GitHub API

```yaml
# scheduled-validation.yml
on:
  schedule: 
    - cron: '0 2 * * *'  # Run daily at 6 AM
```

### Retraining Hook

When drift is detected, the inference service programmatically triggers the CI pipeline

---

## 📦 DVC Integration

Data Version Control (DVC) manages large datasets with Google Drive as remote storage. 

### Setup

```bash
# Install DVC with Google Drive support
pip install dvc dvc-gdrive

# Pull training and testing data
dvc pull 
```

### Authentication

| Environment | Method |
|-------------|--------|
| Local Development | OAuth 2.0 (browser-based) |
| CI/CD Pipeline | Service Account JSON key (GitHub Secret) |

### Integration with CI/CD

The Continuous Delivery workflow hydrates data before training:

```yaml
- name: Setup DVC credentials
  run: |
    echo '${{ secrets.GDRIVE_CREDENTIALS_DATA }}' > credentials.json
    dvc remote modify gdrive --local gdrive_service_account_json_file_path credentials.json

- name: Pull training data
  run: dvc pull training.dvc
```

---

## 🖥 Infrastructure

### Virtual Machines

| VM | Purpose | Resources | Components |
|----|---------|-----------|------------|
| VM1 | Training & MLflow | 3 CPU, 6GB RAM, 50GB | MLflow Server, Training Container, GitHub Runner |
| VM2 | Inference | 1 CPU, 2GB RAM, 50GB | Inference API, GitHub Runner |

### Self-Hosted Runners

GitHub Actions self-hosted runners are installed on both VMs, enabling: 
- Direct pipeline execution without VPN
- Secure communication via GitHub's Long Polling
- Extended workflow timeout (72 hours vs 6 hours)
- Custom resource allocation

### Docker Networks

All containers communicate via the `mlops-net` Docker network:

```yaml
networks:
  mlops-net:
    external: true
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Local Development

```bash
# Clone the repository
git clone https://github.com/AndreFerreira5/downtown-cab-co.git
cd downtown-cab-co

# Install dependencies with uv
uv sync

# Copy environment template
cp .env. example .env
# Edit .env with your MLflow tracking URI and other settings

# Pull data with DVC (requires authentication)
dvc pull

# Run training locally
python -m src.training_api. main
```

### Docker Deployment

```bash
# Create the network
docker network create mlops-net

# Start MLflow server
docker compose -f mlflow.compose.v1.yml up -d

# Start inference API
docker compose -f inference.compose.v1.yml up -d
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `MLFLOW_TRACKING_URI` | MLflow server URL |
| `MLFLOW_MODEL_NAME` | Registered model name |
| `MLFLOW_MODEL_ALIAS` | Model alias (staging/production) |
| `COMMIT_SHA` | Git commit for experiment tagging |
| `MODEL_RETRAINING_TOKEN` | GitHub PAT for workflow dispatch |
| `GITHUB_REPOSITORY` | Repository in `owner/repo` format |

---

## 🛠 Technologies

| Category | Technologies |
|----------|-------------|
| **Languages** | Python 3.11 |
| **ML/Data** | LightGBM, Scikit-learn, Pandas, PyArrow, NumPy |
| **MLOps** | MLflow, DVC |
| **API** | FastAPI, Uvicorn |
| **Containerization** | Docker, Docker Compose |
| **CI/CD** | GitHub Actions, Self-hosted Runners |
| **Registry** | GitHub Container Registry (GHCR) |
| **Storage** | Google Drive (DVC remote) |
| **Testing** | Pytest, httpx |

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 

---

## 📚 References

- [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC User Guide](https://dvc.org/doc/user-guide)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
