"""Unit tests for inference API main module."""

import pytest
from unittest.mock import patch, Mock, MagicMock
from fastapi.testclient import TestClient
import pandas as pd


class TestInferenceAPI:
    """Test cases for inference API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the inference API."""
        with patch('inference_api.main.mlflow.set_tracking_uri'):
            from inference_api.main import app
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
    
    def test_health_endpoint_structure(self, client):
        """Test health endpoint response structure."""
        response = client.get("/health")
        data = response.json()
        
        assert isinstance(data["status"], str)
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["mlflow_uri"], str)
    
    @patch('inference_api.main.load_model_into_app')
    def test_reload_endpoint_success(self, mock_load_model, client):
        """Test successful model reload."""
        mock_load_model.return_value = True
        
        response = client.get("/reload")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"] == "model_reloaded"
        assert "alias" in data
    
    @patch('inference_api.main.load_model_into_app')
    def test_reload_endpoint_failure(self, mock_load_model, client):
        """Test model reload failure."""
        mock_load_model.return_value = False
        
        response = client.get("/reload")
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Failed to load model" in data["detail"]
    
    @patch('inference_api.main.app.state')
    def test_predict_endpoint_model_not_loaded(self, mock_state, client):
        """Test predict endpoint when model is not loaded."""
        mock_state.model = None
        
        request_data = {
            "columns": ["feature1", "feature2"],
            "data": [[1.0, 2.0]]
        }
        
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 503
        data = response.json()
        assert "detail" in data
        assert "Model not loaded" in data["detail"]
    
    @patch('inference_api.main.app.state')
    def test_predict_endpoint_success(self, mock_state, client):
        """Test successful prediction."""
        # Mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = [100.5, 200.3]
        mock_state.model = mock_model
        
        request_data = {
            "columns": ["feature1", "feature2"],
            "data": [[1.0, 2.0], [3.0, 4.0]]
        }
        
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2
        assert data["predictions"][0] == 100.5
        assert data["predictions"][1] == 200.3
    
    @patch('inference_api.main.app.state')
    def test_predict_endpoint_calls_model_predict(self, mock_state, client):
        """Test that predict endpoint calls model.predict with correct data."""
        # Mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = [100.0]
        mock_state.model = mock_model
        
        request_data = {
            "columns": ["col1", "col2", "col3"],
            "data": [[1.0, 2.0, 3.0]]
        }
        
        client.post("/predict", json=request_data)
        
        # Verify model.predict was called
        mock_model.predict.assert_called_once()
        
        # Verify DataFrame was created correctly
        call_args = mock_model.predict.call_args[0][0]
        assert isinstance(call_args, pd.DataFrame)
        assert list(call_args.columns) == ["col1", "col2", "col3"]
        assert len(call_args) == 1
    
    @patch('inference_api.main.app.state')
    def test_predict_endpoint_multiple_rows(self, mock_state, client):
        """Test prediction with multiple rows."""
        # Mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = [10.0, 20.0, 30.0]
        mock_state.model = mock_model
        
        request_data = {
            "columns": ["feature1", "feature2"],
            "data": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        }
        
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 3
    
    def test_predict_endpoint_requires_columns(self, client):
        """Test that predict endpoint requires columns field."""
        request_data = {
            "data": [[1.0, 2.0]]
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_predict_endpoint_requires_data(self, client):
        """Test that predict endpoint requires data field."""
        request_data = {
            "columns": ["feature1", "feature2"]
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422  # Validation error
    
    @patch('inference_api.main.app.state')
    def test_predict_endpoint_returns_floats(self, mock_state, client):
        """Test that predict endpoint returns predictions as floats."""
        # Mock model returning numpy types
        import numpy as np
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([100.5, 200.3])
        mock_state.model = mock_model
        
        request_data = {
            "columns": ["feature1"],
            "data": [[1.0], [2.0]]
        }
        
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify all predictions are Python floats (JSON serializable)
        for pred in data["predictions"]:
            assert isinstance(pred, float)


class TestLoadModelIntoApp:
    """Test cases for load_model_into_app function."""
    
    @patch('inference_api.main.mlflow.pyfunc.load_model')
    def test_load_model_into_app_success(self, mock_load_model):
        """Test successful model loading."""
        from inference_api.main import app, load_model_into_app
        
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        result = load_model_into_app()
        
        assert result is True
        assert app.state.model is mock_model
    
    @patch('inference_api.main.mlflow.pyfunc.load_model')
    def test_load_model_into_app_failure(self, mock_load_model):
        """Test model loading failure."""
        from inference_api.main import app, load_model_into_app
        
        mock_load_model.side_effect = Exception("Model not found")
        
        result = load_model_into_app()
        
        assert result is False
        assert app.state.model is None
    
    @patch('inference_api.main.mlflow.pyfunc.load_model')
    def test_load_model_into_app_uses_correct_uri(self, mock_load_model):
        """Test that model loading uses correct URI format."""
        from inference_api.main import load_model_into_app
        
        mock_load_model.return_value = MagicMock()
        
        load_model_into_app()
        
        # Should use models:/{MODEL_NAME}@{MODEL_ALIAS} format
        call_args = mock_load_model.call_args
        assert 'model_uri' in call_args[1]
        assert 'models:/' in call_args[1]['model_uri']
        assert '@' in call_args[1]['model_uri']
