# ADR-001: LangGraph Pipeline

**Status:** Accepted
**Date:** 2026-04-04

## Context

The RAG pipeline started as a simple LangChain chain: retrieve, rerank, generate. Adding safety layers (injection guard, PII scrubber) required conditional routing: blocked queries should skip generation entirely, not flow through the full chain.

LangChain's sequential chain model doesn't support conditional branching without awkward workarounds (custom chain classes, manual if/else wrappers).

## Decision

Use LangGraph `StateGraph` for the full pipeline (`/ask-graph`) with built-in safety nodes. The basic pipeline (`/ask`, `/ask/stream`) uses router-level injection guards instead of graph nodes, providing the same protection with less overhead.

Key design choices:

1. **Conditional edges** for safety routing. `injection_guard` routes to `END` if the query is unsafe, using `add_conditional_edges`. This is the native LangGraph pattern.

2. **Factory function with closures** for dependency injection. `create_rag_graph(retriever, generator, ...)` returns a compiled graph where each node function closes over its injected dependency. This avoids polluting LangGraph state with infrastructure objects.

3. **Typed state** via `RAGState(TypedDict)` with fields: `query`, `chunks`, `answer`, `use_rerank`, `metadata`, `is_safe`, `error`.

## Consequences

**Good:**
- Safety routing is declarative and visible in the graph structure
- Each node is independently testable
- New nodes (e.g., query expansion) can be added without touching existing nodes
- Graph structure is visible in the ASCII diagrams in `docs/architecture.md`

**Trade-off:**
- The graph pipeline (`/ask-graph`) and basic pipeline (`/ask`) duplicate retrieval/generation logic. `RAGService` handles the basic path, graph nodes handle the graph path. Acceptable at current scale.

**Risk:**
- We intentionally use `ainvoke()` (not `astream()`) because the graph pipeline needs the complete state before streaming the response. `astream()` yields partial node-keyed dicts, not the full accumulated state. See ADR-004 for the simulated streaming approach.
