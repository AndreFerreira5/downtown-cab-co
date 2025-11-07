"""Unit tests for config module."""

import pytest
import os
from unittest.mock import patch, Mock
from src.training_api.config import configure_mlflow


class TestConfigureMlflow:
    """Test cases for configure_mlflow function."""
    
    @patch('src.training_api.config.mlflow.set_tracking_uri')
    @patch.dict(os.environ, {'MLFLOW_TRACKING_URI': 'http://custom-mlflow:5000'})
    def test_configure_mlflow_with_env_var(self, mock_set_tracking_uri):
        """Test configure_mlflow with environment variable set."""
        configure_mlflow()
        
        mock_set_tracking_uri.assert_called_once_with('http://custom-mlflow:5000')
    
    @patch('src.training_api.config.mlflow.set_tracking_uri')
    @patch.dict(os.environ, {}, clear=True)
    def test_configure_mlflow_without_env_var(self, mock_set_tracking_uri):
        """Test configure_mlflow without environment variable (uses default)."""
        configure_mlflow()
        
        mock_set_tracking_uri.assert_called_once_with('http://mlflow:3030')
    
    @patch('src.training_api.config.mlflow.set_tracking_uri')
    @patch.dict(os.environ, {'MLFLOW_TRACKING_URI': ''})
    def test_configure_mlflow_with_empty_env_var(self, mock_set_tracking_uri):
        """Test configure_mlflow with empty environment variable (uses empty string)."""
        configure_mlflow()
        
        mock_set_tracking_uri.assert_called_once_with('')
    
    @patch('src.training_api.config.mlflow.set_tracking_uri')
    @patch.dict(os.environ, {'MLFLOW_TRACKING_URI': 'http://localhost:5001'})
    def test_configure_mlflow_with_localhost(self, mock_set_tracking_uri):
        """Test configure_mlflow with localhost URI."""
        configure_mlflow()
        
        mock_set_tracking_uri.assert_called_once_with('http://localhost:5001')
