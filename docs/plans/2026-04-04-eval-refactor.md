# Eval Refactor — Unified Evaluator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove duplication between RAGEvaluator and RAGEvaluatorWithRerank by merging rerank logic into the base class, creating a single entry point for all eval runs, and adding tests.

**Architecture:** Merge `use_rerank` and `reranker` params into `RAGEvaluator.evaluate()`. Merge `run_evaluation_with_rerank()` into `run_evaluation()`. Delete `run_eval_with_rerank.py` entirely. Fix broken singleton imports (`get_retriever`/`get_reranker` were deleted in DI refactoring) by constructing instances directly.

**Tech Stack:** Python 3.11, pytest, MLflow, sentence-transformers

**Eng Review:** CLEARED — 1 issue found (broken imports), 1 edge case added (empty golden set guard).

---

## Task 1: Write Tests for Pure Metric Functions

**Files:**
- Create: `app/tests/test_eval.py`

**Step 1: Write the tests**

```python
"""Tests for evaluation metrics"""
import pytest
from eval.metrics import calculate_recall_at_k, calculate_mrr


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
```

**Step 2: Run tests to verify they pass**

These test the existing functions which are already implemented. They should pass immediately.

Run: `cd /Users/anton/WorkProjects/pet_projects/mini-wiki-qa && docker compose exec api python -m pytest tests/test_eval.py -v`
Expected: 9 tests PASS

**Step 3: Commit**

```bash
git add app/tests/test_eval.py
git commit -m "test: add tests for calculate_recall_at_k and calculate_mrr"
```

---

## Task 2: Add Empty Golden Set Guard to RAGEvaluator

**Files:**
- Modify: `app/eval/metrics.py:90-91`
- Modify: `app/tests/test_eval.py` (add test)

**Step 1: Write the failing test**

Add to `app/tests/test_eval.py`:

```python
from unittest.mock import MagicMock
from eval.metrics import RAGEvaluator
import json
import tempfile
import os


@pytest.fixture
def golden_set_file():
    """Create a temporary golden set file for testing"""
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
    """Create an empty golden set file"""
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
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/anton/WorkProjects/pet_projects/mini-wiki-qa && docker compose exec api python -m pytest tests/test_eval.py::TestRAGEvaluator::test_evaluate_empty_golden_set -v`
Expected: FAIL with ZeroDivisionError

**Step 3: Add the guard to metrics.py**

In `app/eval/metrics.py`, add guard at the top of `evaluate()` method, after the `samples` line (after line 90):

```python
        samples = self.golden_set[:sample_size] if sample_size else self.golden_set
        logger.info(f"Evaluating on {len(samples)} samples with top_k={top_k}")

        if not samples:
            return {
                "recall@3": 0.0,
                "recall@5": 0.0,
                "mrr": 0.0,
                "avg_latency_ms": 0.0,
                "total_samples": 0,
                "top_k": top_k,
            }
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/anton/WorkProjects/pet_projects/mini-wiki-qa && docker compose exec api python -m pytest tests/test_eval.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add app/eval/metrics.py app/tests/test_eval.py
git commit -m "fix: guard against empty golden set in RAGEvaluator.evaluate()"
```

---

## Task 3: Merge Rerank Logic into RAGEvaluator

**Files:**
- Modify: `app/eval/metrics.py:70-134` (rewrite evaluate method)
- Modify: `app/tests/test_eval.py` (add rerank tests)

**Step 1: Write the failing tests**

Add to `TestRAGEvaluator` class in `app/tests/test_eval.py`:

```python
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
        # Should retrieve 20 candidates for reranking
        mock_retriever.retrieve.assert_called_with(query="What is FastAPI?", top_k=20)
        # Should rerank to top_k=5
        mock_reranker.rerank.assert_called()
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/anton/WorkProjects/pet_projects/mini-wiki-qa && docker compose exec api python -m pytest tests/test_eval.py::TestRAGEvaluator::test_evaluate_with_rerank -v`
Expected: FAIL — `evaluate()` doesn't accept `use_rerank` or `reranker` params

**Step 3: Rewrite evaluate() to support reranking**

Replace the `evaluate` method in `app/eval/metrics.py` (lines 70-134) with:

```python
    def evaluate(
            self,
            retriever,
            top_k: int = 5,
            sample_size: int = None,
            use_rerank: bool = False,
            reranker=None
    ) -> Dict:
        """
        Evaluate retrieval performance with optional reranking

        Args:
            retriever: DocumentRetriever instance
            top_k: Number of documents to retrieve (final count)
            sample_size: Number of samples to evaluate (None = all)
            use_rerank: Whether to apply reranking
            reranker: DocumentReranker instance (required if use_rerank=True)

        Returns:
            Dict with metrics
        """
        import time

        samples = self.golden_set[:sample_size] if sample_size else self.golden_set
        logger.info(f"Evaluating on {len(samples)} samples with top_k={top_k}, rerank={use_rerank}")

        if not samples:
            return {
                "recall@3": 0.0,
                "recall@5": 0.0,
                "mrr": 0.0,
                "avg_latency_ms": 0.0,
                "total_samples": 0,
                "top_k": top_k,
                "use_rerank": use_rerank,
            }

        recall_at_3 = []
        recall_at_5 = []
        mrr_scores = []
        latencies = []

        for idx, item in enumerate(samples):
            query = item["question"]
            ground_truth_doc = item["document"]

            start_time = time.time()

            if use_rerank:
                chunks = retriever.retrieve(query, top_k=20)
            else:
                chunks = retriever.retrieve(query, top_k=top_k)

            if use_rerank and reranker:
                chunks = reranker.rerank(query, chunks, top_k=top_k)

            latency = time.time() - start_time

            retrieved_docs = [
                chunk["source"].split("/")[-1]
                for chunk in chunks
            ]

            recall_at_3.append(calculate_recall_at_k(retrieved_docs, ground_truth_doc, k=3))
            recall_at_5.append(calculate_recall_at_k(retrieved_docs, ground_truth_doc, k=5))
            mrr_scores.append(calculate_mrr(retrieved_docs, ground_truth_doc))
            latencies.append(latency)

            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{len(samples)} samples")

        results = {
            "recall@3": sum(recall_at_3) / len(recall_at_3),
            "recall@5": sum(recall_at_5) / len(recall_at_5),
            "mrr": sum(mrr_scores) / len(mrr_scores),
            "avg_latency_ms": sum(latencies) / len(latencies) * 1000,
            "total_samples": len(samples),
            "top_k": top_k,
            "use_rerank": use_rerank,
        }

        logger.info(f"Evaluation complete: {results}")
        return results
```

**Step 4: Run all tests**

Run: `cd /Users/anton/WorkProjects/pet_projects/mini-wiki-qa && docker compose exec api python -m pytest tests/test_eval.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add app/eval/metrics.py app/tests/test_eval.py
git commit -m "refactor: merge rerank logic into RAGEvaluator.evaluate()"
```

---

## Task 4: Unify run_evaluation() and Delete Duplicate File

**Files:**
- Modify: `app/eval/run_eval.py` (rewrite to support reranking)
- Delete: `app/eval/run_eval_with_rerank.py`

**Step 1: Rewrite run_eval.py**

Replace the entire contents of `app/eval/run_eval.py`:

```python
"""Run evaluation with MLflow tracking"""
import logging
import mlflow
from pathlib import Path

from eval.metrics import RAGEvaluator
from rag.retrieval import DocumentRetriever
from rag.reranker import DocumentReranker
from core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_evaluation(
        golden_set_path: str = "/data/golden_set/squad_qa.json",
        top_k: int = 5,
        sample_size: int = None,
        use_rerank: bool = False,
        experiment_name: str = None,
):
    """
    Run evaluation and log to MLflow

    Args:
        golden_set_path: Path to golden set
        top_k: Number of documents to retrieve
        sample_size: Number of samples (None = all)
        use_rerank: Whether to apply reranking
        experiment_name: MLflow experiment name (auto-generated if None)
    """
    if experiment_name is None:
        experiment_name = "rag-with-rerank" if use_rerank else "rag-baseline"

    logger.info(f"Starting evaluation (rerank={use_rerank})...")

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    retriever = DocumentRetriever()
    reranker = DocumentReranker() if use_rerank else None

    evaluator = RAGEvaluator(golden_set_path)

    with mlflow.start_run():
        mlflow.log_param("top_k", top_k)
        mlflow.log_param("chunk_size", settings.CHUNK_SIZE)
        mlflow.log_param("chunk_overlap", settings.CHUNK_OVERLAP)
        mlflow.log_param("embedding_model", settings.EMBEDDING_MODEL)
        mlflow.log_param("use_rerank", use_rerank)
        mlflow.log_param("sample_size", sample_size or len(evaluator.golden_set))

        results = evaluator.evaluate(
            retriever=retriever,
            top_k=top_k,
            sample_size=sample_size,
            use_rerank=use_rerank,
            reranker=reranker,
        )

        mlflow.log_metric("recall_at_3", results["recall@3"])
        mlflow.log_metric("recall_at_5", results["recall@5"])
        mlflow.log_metric("mrr", results["mrr"])
        mlflow.log_metric("avg_latency_ms", results["avg_latency_ms"])

        results_path = Path("/tmp/eval_results.json")
        import json
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        mlflow.log_artifact(str(results_path))

        logger.info("Evaluation complete!")
        logger.info(f"  Recall@3: {results['recall@3']:.3f}")
        logger.info(f"  Recall@5: {results['recall@5']:.3f}")
        logger.info(f"  MRR: {results['mrr']:.3f}")
        logger.info(f"  Avg Latency: {results['avg_latency_ms']:.1f}ms")

        return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerank", action="store_true", help="Enable reranking")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()

    run_evaluation(
        top_k=args.top_k,
        sample_size=args.samples,
        use_rerank=args.rerank,
    )
```

**Step 2: Delete run_eval_with_rerank.py**

```bash
git rm app/eval/run_eval_with_rerank.py
```

**Step 3: Verify import works**

Run: `cd /Users/anton/WorkProjects/pet_projects/mini-wiki-qa/app && python3 -c "import ast; ast.parse(open('eval/run_eval.py').read()); print('syntax OK')"`
Expected: `syntax OK`

**Step 4: Run all tests**

Run: `cd /Users/anton/WorkProjects/pet_projects/mini-wiki-qa && docker compose exec api python -m pytest tests/test_eval.py -v`
Expected: All tests PASS (no test imports run_eval_with_rerank)

**Step 5: Commit**

```bash
git add app/eval/run_eval.py
git rm app/eval/run_eval_with_rerank.py
git commit -m "refactor: unify run_evaluation() with rerank support, delete duplicate file"
```

---

## Task 5: Docker Build Verification

**Step 1: Build the container**

Run: `docker compose build api`
Expected: Build succeeds

**Step 2: Run all eval tests**

Run: `docker compose exec api python -m pytest tests/test_eval.py -v`
Expected: All tests PASS

**Step 3: Run full test suite**

Run: `docker compose exec api python -m pytest tests/ -v --ignore=tests/test_integration.py`
Expected: All tests PASS (eval tests + existing DI refactor tests)

**Step 4: Verify no references to deleted code**

Run: `grep -r "RAGEvaluatorWithRerank\|run_eval_with_rerank\|get_retriever\|get_reranker" app/eval/ || echo "No stale references"`
Expected: `No stale references`
