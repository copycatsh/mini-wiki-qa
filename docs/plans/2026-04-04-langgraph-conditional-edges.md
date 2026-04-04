# LangGraph Conditional Edges Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace scattered `is_safe` guard checks in graph node functions with a single LangGraph conditional edge after injection_guard, making routing declarative.

**Architecture:** Add `add_conditional_edges()` after injection_guard node: safe queries route to retrieve, unsafe queries route to END. Remove all `is_safe` guard blocks from retrieve_node, rerank_node, generate_node. Keep pii_scrubber_node's `if not state.get("answer")` guard (different concern — handles empty answers, not safety routing).

**Tech Stack:** LangGraph 0.3 (`StateGraph`, `END`, `add_conditional_edges`), pytest

**Eng Review:** CLEARED — 0 issues found.

---

## Task 1: Write Graph Tests

**Files:**
- Create: `app/tests/test_graph.py`

**Step 1: Write all three tests**

```python
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
```

**Step 2: Run tests to verify they pass with current code**

The current graph.py has is_safe guards that make these tests pass even with linear edges (the guards skip nodes). The tests should PASS now — they're testing behavior, not implementation.

Run: `docker compose exec api python -m pytest tests/test_graph.py -v`
Expected: 3 tests PASS

**Step 3: Commit**

```bash
git add app/tests/test_graph.py
git commit -m "test: add graph pipeline tests for safe and unsafe query routing"
```

---

## Task 2: Replace Guards with Conditional Edge

**Files:**
- Modify: `app/rag/graph.py`

**Step 1: Remove is_safe guards from node functions**

In `retrieve_node` (lines 59-61), remove:
```python
        if not state.get("is_safe", True):
            logger.info("Retrieve node: skipped (query blocked)")
            return state
```

In `rerank_node` (lines 79-81), remove:
```python
        if not state.get("is_safe", True):
            logger.info("Rerank node: skipped (query blocked)")
            return state
```

In `generate_node` (lines 102-104), remove:
```python
        if not state.get("is_safe", True):
            logger.info("Generate node: skipped (query blocked)")
            return state
```

Do NOT remove the guard in `pii_scrubber_node` (line 117) — that checks `if not state.get("answer")` which is a different concern (handles empty answers).

**Step 2: Replace linear edge with conditional edge**

Replace this line:
```python
    workflow.add_edge("injection_guard", "retrieve")
```

With:
```python
    def route_after_guard(state: RAGState) -> str:
        return "retrieve" if state["is_safe"] else END

    workflow.add_conditional_edges(
        "injection_guard",
        route_after_guard,
        {"retrieve": "retrieve", END: END},
    )
```

Keep all other edges unchanged:
```python
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", "pii_scrubber")
    workflow.add_edge("pii_scrubber", END)
```

**Step 3: Run tests to verify they still pass**

Run: `docker compose exec api python -m pytest tests/test_graph.py -v`
Expected: 3 tests PASS (behavior unchanged, implementation cleaner)

**Step 4: Run full test suite**

Run: `docker compose exec api python -m pytest tests/ -v --ignore=tests/test_integration.py`
Expected: All tests PASS

**Step 5: Verify no is_safe guards remain in node functions**

Run: `grep -n "is_safe" app/rag/graph.py`
Expected: Only matches in injection_guard_node (setting is_safe) and route_after_guard (reading is_safe). No matches inside retrieve_node, rerank_node, or generate_node.

**Step 6: Commit**

```bash
git add app/rag/graph.py
git commit -m "refactor: replace is_safe guards with LangGraph conditional edge"
```

---

## Task 3: Docker Build Verification

**Step 1: Build**

Run: `docker compose build api`
Expected: Build succeeds

**Step 2: Run full test suite in container**

Run: `docker compose exec api python -m pytest tests/ -v --ignore=tests/test_integration.py`
Expected: All tests PASS
