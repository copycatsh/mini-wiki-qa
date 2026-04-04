# ADR-002: FastAPI Lifespan Dependency Injection

**Status:** Accepted
**Date:** 2026-04-04

## Context

RAG components (retriever, generator, reranker, safety modules) are expensive to initialize: loading embedding models, downloading cross-encoder weights, connecting to Qdrant. They must be initialized once at startup, not per-request.

FastAPI offers two patterns for startup initialization:
1. `@app.on_event("startup")` / `@app.on_event("shutdown")` (deprecated)
2. `lifespan` context manager (recommended since FastAPI 0.95+)

## Decision

Use the `lifespan` async context manager to create all components as singletons on `app.state`. Expose them to route handlers via `Depends()` factory functions in `dependencies.py`.

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.retriever = DocumentRetriever()
    app.state.generator = AnswerGenerator()
    # ...
    yield
    # cleanup if needed
```

```python
def get_retriever(request: Request) -> DocumentRetriever:
    return request.app.state.retriever
```

## Consequences

**Good:**
- Single initialization, shared across all requests
- Clean shutdown path (yield-based context manager)
- Testable: `app.dependency_overrides[get_retriever] = lambda: mock` replaces any component in tests
- Route handlers declare their dependencies explicitly

**Trade-off:**
- All components load at startup, even if some endpoints don't need all of them. Acceptable because startup time (~12s) is a one-time cost.
- `app.state` is untyped. Mitigated by the `Depends()` factories which provide type hints.
