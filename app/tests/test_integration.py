"""Integration test — real Qdrant (in-memory) + real embeddings

This test uses real components (not mocks) to verify DI wiring end-to-end.
Requires langchain packages — runs in Docker only.
"""
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

from api.main import app
from api.dependencies import get_retriever, get_generator, get_reranker, get_rag_graph
from core.config import settings


@pytest.fixture(scope="module")
def embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@pytest.fixture(scope="module")
def qdrant_store(embeddings):
    client = QdrantClient(":memory:")
    collection_name = "test-integration"

    vector_size = len(embeddings.embed_query("test"))
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    docs = [
        Document(page_content="Python is a programming language created by Guido van Rossum.", metadata={"source": "python.md"}),
        Document(page_content="FastAPI is a modern web framework for building APIs with Python.", metadata={"source": "fastapi.md"}),
    ]

    store = QdrantVectorStore(client=client, collection_name=collection_name, embedding=embeddings)
    store.add_documents(docs)
    return store


@pytest.fixture
def real_retriever(qdrant_store, embeddings):
    retriever = MagicMock()

    def real_retrieve(query, top_k=5):
        results = qdrant_store.similarity_search_with_score(query, k=top_k)
        return [
            {"text": doc.page_content, "source": doc.metadata.get("source", "unknown"), "score": float(score)}
            for doc, score in results
        ]

    retriever.retrieve = real_retrieve
    retriever.embeddings = embeddings
    return retriever


@pytest.fixture
def integration_client(real_retriever):
    mock_generator = MagicMock()
    mock_generator.generate.return_value = "Python is a programming language."
    mock_reranker = MagicMock()
    mock_graph = MagicMock()

    app.dependency_overrides[get_retriever] = lambda: real_retriever
    app.dependency_overrides[get_generator] = lambda: mock_generator
    app.dependency_overrides[get_reranker] = lambda: mock_reranker
    app.dependency_overrides[get_rag_graph] = lambda: mock_graph

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


@pytest.mark.slow
def test_ask_end_to_end(integration_client):
    resp = integration_client.post(
        "/ask",
        json={"query": "What is Python?"},
        headers={"X-API-Key": settings.API_SHARED_SECRET},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert len(data["citations"]) > 0
    assert any("python" in c["text"].lower() for c in data["citations"])
