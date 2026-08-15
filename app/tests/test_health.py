"""Tests for health and root endpoints"""


def test_root_returns_info(client):
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["message"] == "Rag Playground API"
    assert "docs" in data
    assert "health" in data


def test_health_check_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "timestamp" in data
    assert "services" in data
