"""Tests for multi-query retrieval utility functions"""
import pytest
from unittest.mock import MagicMock

from rag.retrieval import expand_queries, deduplicate_chunks, multi_query_retrieve


# --- deduplicate_chunks ---


class TestDeduplicateChunks:
    def test_keeps_highest_score(self):
        chunks = [
            {"text": "same content", "source": "a.md", "score": 0.85},
            {"text": "same content", "source": "a.md", "score": 0.92},
        ]
        result = deduplicate_chunks(chunks, top_k=5)
        assert len(result) == 1
        assert result[0]["score"] == 0.92

    def test_all_identical(self):
        chunks = [
            {"text": "identical", "source": "a.md", "score": 0.5},
            {"text": "identical", "source": "a.md", "score": 0.6},
            {"text": "identical", "source": "a.md", "score": 0.7},
        ]
        result = deduplicate_chunks(chunks, top_k=5)
        assert len(result) == 1
        assert result[0]["score"] == 0.7

    def test_no_duplicates(self):
        chunks = [
            {"text": "chunk A", "source": "a.md", "score": 0.9},
            {"text": "chunk B", "source": "b.md", "score": 0.8},
            {"text": "chunk C", "source": "c.md", "score": 0.7},
        ]
        result = deduplicate_chunks(chunks, top_k=5)
        assert len(result) == 3

    def test_empty_input(self):
        result = deduplicate_chunks([], top_k=5)
        assert result == []

    def test_trims_to_top_k(self):
        chunks = [
            {"text": f"chunk {i}", "source": "a.md", "score": i * 0.1}
            for i in range(10)
        ]
        result = deduplicate_chunks(chunks, top_k=5)
        assert len(result) == 5
        assert result[0]["score"] > result[1]["score"]


# --- expand_queries ---


class TestExpandQueries:
    def _mock_llm(self, response_text: str) -> MagicMock:
        llm = MagicMock()
        resp = MagicMock()
        resp.content = response_text
        llm.invoke.return_value = resp
        return llm

    def test_happy_path(self):
        llm = self._mock_llm("What is Python used for?\nHow is Python applied?\nWhere is Python used?")
        result = expand_queries(llm, "What is Python?", num_queries=3)
        assert result[0] == "What is Python?"
        assert len(result) == 4

    def test_empty_response(self):
        llm = self._mock_llm("")
        result = expand_queries(llm, "What is Python?", num_queries=3)
        assert result == ["What is Python?"]

    def test_numbered_lines(self):
        llm = self._mock_llm("1. What is Python used for?\n2. How is Python applied?\n3. Where is Python used?")
        result = expand_queries(llm, "What is Python?", num_queries=3)
        assert result[0] == "What is Python?"
        assert "1." not in result[1]
        assert result[1] == "What is Python used for?"
        assert len(result) == 4

    def test_with_bullets(self):
        llm = self._mock_llm("- What is Python used for?\n- How is Python applied?")
        result = expand_queries(llm, "What is Python?", num_queries=2)
        assert result[0] == "What is Python?"
        assert result[1] == "What is Python used for?"
        assert len(result) == 3

    def test_includes_original(self):
        llm = self._mock_llm("variant one\nvariant two")
        result = expand_queries(llm, "original query", num_queries=2)
        assert result[0] == "original query"


# --- multi_query_retrieve ---


class TestMultiQueryRetrieve:
    def _mock_llm(self, response_text: str) -> MagicMock:
        llm = MagicMock()
        resp = MagicMock()
        resp.content = response_text
        llm.invoke.return_value = resp
        return llm

    def test_happy_path(self):
        llm = self._mock_llm("variant A\nvariant B")
        retriever = MagicMock()
        retriever.retrieve.side_effect = [
            [{"text": "chunk 1", "source": "a.md", "score": 0.9}],
            [{"text": "chunk 2", "source": "b.md", "score": 0.8}],
            [{"text": "chunk 1", "source": "a.md", "score": 0.85}],
        ]
        result = multi_query_retrieve(llm, retriever, "test query", top_k=5, num_queries=2)
        assert retriever.retrieve.call_count == 3
        assert len(result) == 2
        assert result[0]["score"] == 0.9
        assert result[1]["score"] == 0.8

    def test_fallback_on_llm_error(self):
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("LLM failed")
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            {"text": "fallback chunk", "source": "a.md", "score": 0.7}
        ]
        result = multi_query_retrieve(llm, retriever, "test query", top_k=5)
        assert len(result) == 1
        assert result[0]["text"] == "fallback chunk"
        retriever.retrieve.assert_called_once_with("test query", top_k=5)

    def test_partial_retrieval_failure(self):
        """One variant's retrieval fails, others succeed — partial results used."""
        llm = self._mock_llm("variant A\nvariant B")
        retriever = MagicMock()
        retriever.retrieve.side_effect = [
            [{"text": "chunk 1", "source": "a.md", "score": 0.9}],
            RuntimeError("Qdrant timeout"),
            [{"text": "chunk 2", "source": "b.md", "score": 0.8}],
        ]
        result = multi_query_retrieve(llm, retriever, "test query", top_k=5, num_queries=2)
        assert len(result) == 2
        assert result[0]["score"] == 0.9

    def test_all_retrievals_fail_falls_back(self):
        """All variant retrievals fail — falls back to single query."""
        llm = self._mock_llm("variant A")
        retriever = MagicMock()
        retriever.retrieve.side_effect = [
            RuntimeError("fail 1"),
            RuntimeError("fail 2"),
            [{"text": "fallback", "source": "a.md", "score": 0.5}],
        ]
        result = multi_query_retrieve(llm, retriever, "test query", top_k=5, num_queries=1)
        assert len(result) == 1
        assert result[0]["text"] == "fallback"
