"""Shared test fixtures"""
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from api.main import app
from api.dependencies import (
    verify_api_key, get_retriever, get_generator,
    get_reranker, get_pii_scrubber, get_injection_guard, get_rag_graph,
)
from core.config import settings

VALID_API_KEY = settings.API_SHARED_SECRET


@pytest.fixture
def mock_retriever():
    m = MagicMock()
    m.retrieve.return_value = [
        {"text": "Test chunk content", "source": "test/doc.md", "score": 0.95}
    ]
    m.embeddings = MagicMock()
    return m


@pytest.fixture
def mock_generator():
    m = MagicMock()
    m.generate.return_value = "This is a test answer."
    return m


@pytest.fixture
def mock_reranker():
    m = MagicMock()
    m.rerank.return_value = [
        {"text": "Test chunk content", "source": "test/doc.md", "score": 0.95, "rerank_score": 0.99}
    ]
    return m


@pytest.fixture
def mock_pii_scrubber():
    m = MagicMock()
    m.scrub.return_value = {"text": "clean answer", "pii_detected": [], "was_scrubbed": False}
    return m


@pytest.fixture
def mock_injection_guard():
    m = MagicMock()
    m.check.return_value = {"is_safe": True, "detected_patterns": [], "risk_level": "none"}
    return m


@pytest.fixture
def mock_rag_graph():
    m = MagicMock()
    m.invoke.return_value = {
        "query": "test",
        "chunks": [{"text": "Test chunk", "source": "test/doc.md", "score": 0.9}],
        "answer": "Graph answer",
        "use_rerank": False,
        "metadata": {"injection_check": {"is_safe": True, "risk_level": "none"}},
        "is_safe": True,
        "error": "",
    }
    return m


@pytest.fixture
def client(mock_retriever, mock_generator, mock_reranker, mock_pii_scrubber, mock_injection_guard, mock_rag_graph):
    app.dependency_overrides[get_retriever] = lambda: mock_retriever
    app.dependency_overrides[get_generator] = lambda: mock_generator
    app.dependency_overrides[get_reranker] = lambda: mock_reranker
    app.dependency_overrides[get_pii_scrubber] = lambda: mock_pii_scrubber
    app.dependency_overrides[get_injection_guard] = lambda: mock_injection_guard
    app.dependency_overrides[get_rag_graph] = lambda: mock_rag_graph

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()
