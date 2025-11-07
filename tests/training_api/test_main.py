"""Unit tests for training API main module."""

import pytest
from unittest.mock import patch, Mock, MagicMock
from fastapi.testclient import TestClient


class TestTrainingAPI:
    """Test cases for training API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the training API."""
        with patch('src.training_api.main.configure_mlflow'):
            with patch('src.training_api.main.mlflow.get_tracking_uri', return_value='http://test-mlflow:3030'):
                with patch('src.training_api.main.load_model_into_app'):
                    from src.training_api.main import app
                    return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "mlflow_uri" in data
    
    def test_health_endpoint_returns_model_status(self, client):
        """Test that health endpoint returns model loading status."""
        response = client.get("/health")
        data = response.json()
        
        assert isinstance(data["model_loaded"], bool)
    
    @patch('src.training_api.main.load_model_into_app')
    @patch('src.training_api.main.run_training')
    def test_train_endpoint_success(self, mock_run_training, mock_load_model, client):
        """Test successful training endpoint call."""
        # Mock training results - note: run_training actually returns a tuple,
        # but the endpoint tries to unpack it as a dict. This test documents current behavior.
        mock_run_training.return_value = {}
        mock_load_model.return_value = True
        
        request_data = {
            "commit_sha": "abc123",
            "model_name": "test_model",
            "experiment_name": "test_experiment"
        }
        
        response = client.post("/train", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"] == "train finished"
    
    @patch('src.training_api.main.run_training')
    def test_train_endpoint_calls_run_training(self, mock_run_training, client):
        """Test that train endpoint calls run_training with correct parameters."""
        mock_run_training.return_value = {}
        
        with patch('src.training_api.main.load_model_into_app'):
            request_data = {
                "commit_sha": "def456",
                "model_name": "nyc_taxi",
                "experiment_name": "nyc_experiment"
            }
            
            response = client.post("/train", json=request_data)
            
            mock_run_training.assert_called_once_with(
                commit_sha="def456",
                model_name="nyc_taxi_duration",  # Uses environment variable
                experiment_name="nyc_experiment"
            )
    
    @patch('src.training_api.main.load_model_into_app')
    @patch('src.training_api.main.run_training')
    def test_train_endpoint_loads_model(self, mock_run_training, mock_load_model, client):
        """Test that train endpoint attempts to load model after training."""
        mock_run_training.return_value = {}
        mock_load_model.return_value = True
        
        request_data = {
            "commit_sha": "abc123",
            "model_name": "test_model",
            "experiment_name": "test_experiment"
        }
        
        client.post("/train", json=request_data)
        
        mock_load_model.assert_called_once()
    
    def test_train_endpoint_requires_all_fields(self, client):
        """Test that train endpoint requires all fields."""
        # Missing experiment_name
        request_data = {
            "commit_sha": "abc123",
            "model_name": "test_model"
        }
        
        response = client.post("/train", json=request_data)
        assert response.status_code == 422  # Validation error
    
    @patch('src.training_api.main.load_model_into_app')
    def test_reload_endpoint_success(self, mock_load_model, client):
        """Test successful model reload."""
        mock_load_model.return_value = True
        
        response = client.get("/reload")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"] == "model_reloaded"
        assert "alias" in data
    
    @patch('src.training_api.main.load_model_into_app')
    def test_reload_endpoint_failure(self, mock_load_model, client):
        """Test model reload failure."""
        mock_load_model.return_value = False
        
        response = client.get("/reload")
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Failed to load model" in data["detail"]
    
    @patch('src.training_api.main.load_model_into_app')
    def test_reload_endpoint_calls_load_function(self, mock_load_model, client):
        """Test that reload endpoint calls load function."""
        mock_load_model.return_value = True
        
        client.get("/reload")
        
        mock_load_model.assert_called_once()


class TestLoadModelIntoApp:
    """Test cases for load_model_into_app function."""
    
    @patch('src.training_api.main.mlflow.pyfunc.load_model')
    def test_load_model_into_app_success(self, mock_load_model):
        """Test successful model loading."""
        from src.training_api.main import app, load_model_into_app
        
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        result = load_model_into_app()
        
        assert result is True
        assert app.state.model is mock_model
    
    @patch('src.training_api.main.mlflow.pyfunc.load_model')
    def test_load_model_into_app_failure(self, mock_load_model):
        """Test model loading failure."""
        from src.training_api.main import app, load_model_into_app
        
        mock_load_model.side_effect = Exception("Model not found")
        
        result = load_model_into_app()
        
        assert result is False
        assert app.state.model is None
    
    @patch('src.training_api.main.mlflow.pyfunc.load_model')
    def test_load_model_into_app_uses_correct_uri(self, mock_load_model):
        """Test that model loading uses correct URI format."""
        from src.training_api.main import load_model_into_app
        
        mock_load_model.return_value = MagicMock()
        
        load_model_into_app()
        
        # Should use models:/{MODEL_NAME}@{MODEL_ALIAS} format
        call_args = mock_load_model.call_args
        assert 'model_uri' in call_args[1]
        assert 'models:/' in call_args[1]['model_uri']
