from fastapi.testclient import TestClient
from api.main import app


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert "status" in j and j["status"] == "healthy"
    assert "model_loaded" in j
    assert "timestamp" in j
