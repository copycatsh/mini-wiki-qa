# Architecture

## System Overview

```
                          ┌─────────────────────────────┐
                          │         Docker Network       │
                          │                              │
  Browser ──► Next.js     │  FastAPI ──► Qdrant          │
              :3000 ──────┤  :8000      :6333            │
                          │    │                         │
                          │    ├──► LM Studio (host)     │
                          │    │    :1234                 │
                          │    ├──► Ollama                │
                          │    │    :11434                │
                          │    ├──► MLflow ──► MinIO      │
                          │    │    :5001      :9000      │
                          │    └──► n8n                   │
                          │         :5678                 │
                          └─────────────────────────────┘
```

## RAG Pipeline

Two pipelines serve the same purpose with different safety guarantees.

### Basic Pipeline

**`/ask`** (sync, with injection guard):

```
  Query + History
    │
    ▼
  InjectionGuard (query + history) ── blocked ──► 400 Error
    │ safe
    ▼
  Retrieve → Rerank (optional) → Generate (with history) → Response
```

**`/ask/stream`** (SSE streaming, with injection guard):

```
  Query + History
    │
    ▼
  InjectionGuard (query + history) ── blocked ──► Error event
    │ safe
    ▼
  Retrieve (Qdrant semantic search)
    │
    ▼
  Rerank (optional, cross-encoder)
    │
    ▼
  Generate (LLM.astream with history → SSE tokens)
    │
    ▼
  Citations + Metadata + Done
```

All endpoints accept an optional `history` field (list of prior user/assistant messages) for conversational context. History is validated by Pydantic (`HistoryMessage` model) and checked by the injection guard before reaching the LLM.

### Graph Pipeline (`/ask-graph`, `/ask-graph/stream`)

```
  Query
    │
    ▼
  ┌─────────────────┐
  │ injection_guard  │── unsafe ──► END (blocked)
  └────────┬────────┘
           │ safe
           ▼
  ┌─────────────────┐
  │    retrieve      │  top_k=20 if rerank, else 5
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │     rerank       │  always visited; skips internally when use_rerank=False
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │    generate      │
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │  pii_scrubber    │  removes PII from answer
  └────────┬────────┘
           ▼
         END
```

Implemented as a LangGraph `StateGraph` with conditional edges. The `injection_guard` node checks both the query and any history messages, routing to `END` if unsafe.

The streaming variant (`/ask-graph/stream`) runs the full graph via `ainvoke()`, then word-splits the completed answer into SSE token events. This is simulated streaming, not real token streaming. See [ADR-004](decisions/004-sse-streaming.md).

## SSE Streaming

```
  Frontend (useChat hook)
    │
    │  POST /ask/stream or /ask-graph/stream
    │  Content-Type: application/json
    │  X-API-Key: ...
    │
    ▼
  FastAPI StreamingResponse
    │
    │  text/event-stream
    │  data: {"event": "token", "data": "Hello"}
    │  data: {"event": "token", "data": " world"}
    │  data: {"event": "citations", "data": [...]}
    │  data: {"event": "metadata", "data": {...}}
    │  data: {"event": "done", "data": ""}
    │
    ▼
  streamAsk() async generator
    │  fetch() + ReadableStream
    │  Manual SSE line parsing
    │
    ▼
  useChat() React hook
    │  Dispatches events to message state
    │  Supports abort via AbortController
```

Event types: `token`, `citations`, `metadata`, `error`, `done`.

## Dependency Injection

FastAPI lifespan creates all singleton components at startup:

```
  lifespan(app)
    ├── app.state.retriever     = DocumentRetriever()
    ├── app.state.generator     = AnswerGenerator()
    ├── app.state.reranker      = DocumentReranker()
    ├── app.state.pii_scrubber  = PIIScrubber()
    ├── app.state.injection_guard = InjectionGuard()
    └── app.state.rag_graph     = create_rag_graph(...)
```

Route handlers receive components via `Depends()`:

```python
@router.post("/ask/stream")
async def ask_stream(
    retriever=Depends(get_retriever),
    generator=Depends(get_generator),
    reranker=Depends(get_reranker),
    injection_guard=Depends(get_injection_guard),
):
    service = RAGService(retriever, generator, reranker)
```

Tests override dependencies via `app.dependency_overrides`. See [ADR-002](decisions/002-fastapi-lifespan-di.md).

## Docker Services

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| api | Custom (app/Dockerfile) | 8000 | FastAPI backend |
| frontend | Custom (frontend/Dockerfile) | 3000 | Next.js chat UI |
| qdrant | qdrant/qdrant:v1.10.0 | 6333, 6334 | Vector database |
| mlflow | ghcr.io/mlflow/mlflow | 5001 | Experiment tracking |
| minio | minio/minio | 9000, 9001 | Object storage (MLflow artifacts) |
| n8n | n8nio/n8n | 5678 | Workflow automation (profile: `automation`) |
| ollama | ollama/ollama | 11434 | LLM server (profile: `production`) |

n8n and ollama use Docker Compose profiles and don't start by default. Activate with `docker compose --profile <name> up`.

LM Studio runs on the host machine (not in Docker) at port 1234 and is accessed via `host.docker.internal`.

## Key Files

```
app/
├── api/
│   ├── main.py             # Lifespan DI, CORS, router registration
│   ├── dependencies.py     # Depends() factories
│   ├── schemas.py          # Pydantic models (AskRequest, HistoryMessage, etc.)
│   ├── utils.py            # Shared router utilities (history serialization, injection checks)
│   └── routers/
│       ├── health.py       # GET /, GET /health
│       ├── ask.py          # POST /ask, POST /ask-graph
│       ├── stream.py       # POST /ask/stream, POST /ask-graph/stream
│       └── admin.py        # POST /ingest
├── services/
│   └── rag_service.py      # Orchestrates retriever + generator + reranker
├── rag/
│   ├── retrieval.py        # Qdrant semantic search
│   ├── generation.py       # LLM answer generation
│   ├── reranker.py         # Cross-encoder reranking
│   ├── ingest.py           # Document ingestion pipeline
│   ├── safety.py           # InjectionGuard, PIIScrubber
│   └── graph.py            # LangGraph RAG pipeline
└── core/
    └── config.py           # Settings from env vars
```
