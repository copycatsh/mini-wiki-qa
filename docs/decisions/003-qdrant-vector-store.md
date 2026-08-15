# ADR-003: Qdrant Vector Store

**Status:** Accepted

## Context

The RAG system needs a vector database for semantic search over document chunks. Options considered:

| Database | Type | Persistence | Filtering | API |
|----------|------|-------------|-----------|-----|
| FAISS | In-memory library | File-based | Limited | Python only |
| Chroma | Embedded DB | SQLite | Metadata | Python, REST |
| Qdrant | Standalone server | On-disk | Rich payload | REST, gRPC |

## Decision

Use Qdrant as a standalone Docker service.

Key factors:
1. **Standalone server** separates concerns. The API container doesn't hold vector state.
2. **Persistent storage** via Docker volume (`qdrant_data`). Survives container restarts.
3. **LangChain integration** via `langchain-qdrant` package for both indexing and retrieval.
4. **Dashboard** at `:6333/dashboard` for inspecting collections and points.

### Version requirement

Qdrant **v1.10.0+** is required. The `langchain-qdrant 0.2.0` package uses the `query_points` API which was introduced in Qdrant 1.10. Earlier versions (e.g., v1.8.4) return 404 on `/collections/{name}/points/query`.

### Collection design

One collection per project (`rag-playground`). Cosine distance. 384-dimensional vectors (matching `all-MiniLM-L6-v2` embeddings).

## Consequences

**Good:**
- Data persists across deploys
- Can inspect/debug vectors via dashboard
- Scales independently from the API
- Rich filtering support for future metadata queries

**Trade-off:**
- Requires running a separate service (more Docker resources). Worth it for persistence and debuggability.
- Collection must be created before first query. The ingestion pipeline handles this via `create_collection()`.
