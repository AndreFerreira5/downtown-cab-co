"""Pytest configuration and shared fixtures."""

import pytest
import sys
import os

# Add src directory to Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables before each test."""
    # Store original environment
    original_env = os.environ.copy()
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_mlflow_uri():
    """Provide a mock MLflow tracking URI."""
    return "http://test-mlflow:5000"
