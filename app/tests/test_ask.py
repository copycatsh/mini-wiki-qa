"""Tests for /ask and /ask-graph endpoints"""
from tests.conftest import VALID_API_KEY


def test_ask_requires_api_key(client):
    resp = client.post("/ask", json={"query": "What is Python?"})
    assert resp.status_code == 422


def test_ask_invalid_api_key(client):
    resp = client.post(
        "/ask",
        json={"query": "What is Python?"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert resp.status_code == 403


def test_ask_happy_path(client, mock_retriever, mock_generator):
    resp = client.post(
        "/ask",
        json={"query": "What is Python?"},
        headers={"X-API-Key": VALID_API_KEY},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert data["answer"] == "This is a test answer."
    assert len(data["citations"]) > 0
    mock_retriever.retrieve.assert_called_once()
    mock_generator.generate.assert_called_once()


def test_ask_with_rerank(client, mock_retriever, mock_reranker, mock_generator):
    resp = client.post(
        "/ask",
        json={"query": "What is Python?", "use_rerank": True},
        headers={"X-API-Key": VALID_API_KEY},
    )
    assert resp.status_code == 200
    mock_reranker.rerank.assert_called_once()


def test_ask_graph_happy_path(client, mock_rag_graph):
    resp = client.post(
        "/ask-graph",
        json={"query": "What is Python?"},
        headers={"X-API-Key": VALID_API_KEY},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "Graph answer"
    assert data["metadata"]["pipeline"] == "langgraph"
    mock_rag_graph.invoke.assert_called_once()
