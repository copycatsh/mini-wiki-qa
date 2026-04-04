"""Tests for LangGraph RAG pipeline — conditional edge routing"""
import pytest
from unittest.mock import MagicMock
from rag.graph import create_rag_graph


@pytest.fixture
def mock_deps():
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        {"text": "Python is great", "source": "python.md", "score": 0.9}
    ]

    generator = MagicMock()
    generator.generate.return_value = "Python is a programming language."

    reranker = MagicMock()

    pii_scrubber = MagicMock()
    pii_scrubber.scrub.return_value = {
        "text": "Python is a programming language.",
        "pii_detected": [],
        "was_scrubbed": False,
    }

    injection_guard = MagicMock()

    return {
        "retriever": retriever,
        "generator": generator,
        "reranker": reranker,
        "pii_scrubber": pii_scrubber,
        "injection_guard": injection_guard,
    }


def _make_graph(mock_deps):
    return create_rag_graph(**mock_deps)


def _base_state():
    return {
        "query": "What is Python?",
        "chunks": [],
        "answer": "",
        "use_rerank": False,
        "metadata": {},
        "is_safe": True,
        "error": "",
    }


class TestGraphSafeQuery:
    def test_safe_query_flows_through_all_nodes(self, mock_deps):
        mock_deps["injection_guard"].check.return_value = {
            "is_safe": True,
            "detected_patterns": [],
            "risk_level": "none",
        }

        graph = _make_graph(mock_deps)
        result = graph.invoke(_base_state())

        assert result["is_safe"] is True
        assert result["answer"] == "Python is a programming language."
        mock_deps["retriever"].retrieve.assert_called_once()
        mock_deps["generator"].generate.assert_called_once()
        mock_deps["pii_scrubber"].scrub.assert_called_once()


class TestGraphUnsafeQuery:
    def test_unsafe_query_stops_at_injection_guard(self, mock_deps):
        mock_deps["injection_guard"].check.return_value = {
            "is_safe": False,
            "detected_patterns": ["ignore previous instructions"],
            "risk_level": "high",
        }

        graph = _make_graph(mock_deps)
        result = graph.invoke(_base_state())

        assert result["is_safe"] is False
        mock_deps["retriever"].retrieve.assert_not_called()
        mock_deps["generator"].generate.assert_not_called()
        mock_deps["pii_scrubber"].scrub.assert_not_called()

    def test_blocked_answer_is_set_correctly(self, mock_deps):
        mock_deps["injection_guard"].check.return_value = {
            "is_safe": False,
            "detected_patterns": ["system:"],
            "risk_level": "high",
        }

        graph = _make_graph(mock_deps)
        result = graph.invoke(_base_state())

        assert "blocked for security reasons" in result["answer"]
        assert result["error"] != ""
        assert result["metadata"]["injection_check"]["is_safe"] is False
