"""Tests for evaluation metrics"""
import pytest
import json
import tempfile
import os
from unittest.mock import MagicMock

from eval.metrics import calculate_recall_at_k, calculate_mrr, RAGEvaluator


class TestCalculateRecallAtK:
    def test_hit_in_top_k(self):
        docs = ["a.md", "b.md", "c.md", "d.md"]
        assert calculate_recall_at_k(docs, "b.md", k=3) is True

    def test_miss_outside_top_k(self):
        docs = ["a.md", "b.md", "c.md", "d.md"]
        assert calculate_recall_at_k(docs, "d.md", k=3) is False

    def test_exact_k_boundary(self):
        docs = ["a.md", "b.md", "c.md"]
        assert calculate_recall_at_k(docs, "c.md", k=3) is True

    def test_not_in_list_at_all(self):
        docs = ["a.md", "b.md"]
        assert calculate_recall_at_k(docs, "z.md", k=5) is False

    def test_empty_retrieved(self):
        assert calculate_recall_at_k([], "a.md", k=3) is False


class TestCalculateMRR:
    def test_first_position(self):
        docs = ["target.md", "b.md", "c.md"]
        assert calculate_mrr(docs, "target.md") == 1.0

    def test_third_position(self):
        docs = ["a.md", "b.md", "target.md"]
        assert calculate_mrr(docs, "target.md") == pytest.approx(1.0 / 3)

    def test_not_found(self):
        docs = ["a.md", "b.md"]
        assert calculate_mrr(docs, "z.md") == 0.0

    def test_empty_list(self):
        assert calculate_mrr([], "a.md") == 0.0


@pytest.fixture
def golden_set_file():
    data = [
        {"question": "What is Python?", "document": "python.md"},
        {"question": "What is FastAPI?", "document": "fastapi.md"},
    ]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def empty_golden_set_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump([], f)
        path = f.name
    yield path
    os.unlink(path)


class TestRAGEvaluator:
    def test_evaluate_empty_golden_set(self, empty_golden_set_file):
        evaluator = RAGEvaluator(empty_golden_set_file)
        mock_retriever = MagicMock()
        result = evaluator.evaluate(retriever=mock_retriever)
        assert result["total_samples"] == 0
        assert result["recall@3"] == 0.0
        assert result["recall@5"] == 0.0
        assert result["mrr"] == 0.0
        mock_retriever.retrieve.assert_not_called()

    def test_evaluate_without_rerank(self, golden_set_file):
        evaluator = RAGEvaluator(golden_set_file)
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            {"source": "path/python.md", "text": "content", "score": 0.9},
            {"source": "path/other.md", "text": "content", "score": 0.8},
        ]

        result = evaluator.evaluate(retriever=mock_retriever, top_k=5)

        assert result["total_samples"] == 2
        assert "recall@3" in result
        assert "mrr" in result
        assert result["use_rerank"] is False
        mock_retriever.retrieve.assert_called_with(query="What is FastAPI?", top_k=5)

    def test_evaluate_with_rerank(self, golden_set_file):
        evaluator = RAGEvaluator(golden_set_file)
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            {"source": "path/python.md", "text": "content", "score": 0.9},
        ] * 20
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            {"source": "path/python.md", "text": "content", "score": 0.9, "rerank_score": 0.95},
        ]

        result = evaluator.evaluate(
            retriever=mock_retriever,
            top_k=5,
            use_rerank=True,
            reranker=mock_reranker,
        )

        assert result["use_rerank"] is True
        mock_retriever.retrieve.assert_called_with(query="What is FastAPI?", top_k=20)
        mock_reranker.rerank.assert_called()
