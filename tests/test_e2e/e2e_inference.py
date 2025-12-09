import pytest
import requests
import pandas as pd
import mlflow
from src.inference_api.main import app
from fastapi.testclient import TestClient


class TestE2EInference:
    """End-to-end tests for the inference pipeline."""

    @pytest.fixture
    def client(self):
        """Test client using FastAPI's TestClient."""
        return TestClient(app, base_url="http://localhost:9001")

    @pytest.fixture
    def sample_prediction_data(self):
        """Sample data that mimics real taxi trip requests.
            Note: this data was created arbitrarily for testing purposes. It is not representative of real data.
        """
        return {
            "data": [
                {
                    "tpep_pickup_datetime": "2013-01-01 00:00:00",
                    "tpep_dropoff_datetime": "2013-01-01 00:20:00",
                    "passenger_count": 2,
                    "trip_distance": 3.5,
                    "PULocationID": 100,
                    "DOLocationID": 200
                }
            ]
        }

    def test_health_endpoint(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_prediction_e2e(self, client, sample_prediction_data):
        """Test full prediction flow end-to-end."""
        response = client.post("/predict", json=sample_prediction_data)

        assert response.status_code == 200
        result = response.json()

        # Verify structure
        assert "predictions" in result
        assert len(result["predictions"]) == 1

        # Verify prediction quality
        prediction = result["predictions"][0]
        assert prediction > 0  # Duration should be positive
        assert prediction < 10800  # Less than 3 hours (reasonable)
        assert not pd.isna(prediction)  # No NaN values

    def test_batch_predictions(self, client):
        """Test batch prediction capability.
            Note: As before, this data was created arbitrarily for testing purposes.
        """
        batch_data = {
            "data": [
                {
                    "tpep_pickup_datetime": f"2013-01-01 {i:02d}:00:00",
                    "tpep_dropoff_datetime": f"2013-01-01 {i: 02d}:20:00",
                    "passenger_count": i % 5 + 1,
                    "trip_distance": 2.0 + i * 0.5,
                    "PULocationID": 100 + i,
                    "DOLocationID": 200 + i
                }
                for i in range(10)
            ]
        }

        response = client.post("/predict", json=batch_data)
        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 10

    def test_invalid_input_handling(self, client):
        """Test error handling for invalid inputs."""
        invalid_data = {"data": [{"invalid_field": "test"}]}

        response = client.post("/predict", json=invalid_data)
        # Should handle gracefully (400 or process with defaults)
        assert response.status_code in [400, 200]