from fastapi.testclient import TestClient

from src.training_api.main import app

client = TestClient(app)


def test_ping_pong():
    response = client.get("/health")
    assert response.status_code == 200
