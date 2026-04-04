"""Tests for /ingest endpoint"""
from unittest.mock import patch, MagicMock
from tests.conftest import VALID_API_KEY


def test_ingest_requires_api_key(client):
    resp = client.post("/ingest")
    assert resp.status_code == 422


def test_ingest_invalid_api_key(client):
    resp = client.post("/ingest", headers={"X-API-Key": "wrong"})
    assert resp.status_code == 403
