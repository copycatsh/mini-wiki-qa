# Mini-Wiki Q&A

RAG-powered question answering over local documents. Semantic search, reranking, LangGraph safety pipeline, SSE streaming, and a chat UI.

## Quick Start

```bash
make setup          # creates .env from template
# edit .env: set LLM_BACKEND, API_SHARED_SECRET
make all            # builds and starts all services
```

Open http://localhost:3000 for the chat UI, or http://localhost:8000/docs for the API.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI, Python 3.11+ |
| Frontend | Next.js 14, React 18, shadcn/ui, Tailwind CSS |
| LLM Orchestration | LangChain, LangGraph |
| Vector DB | Qdrant v1.10.0 |
| LLM Backend | LM Studio (dev) / Ollama (prod) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Experiment Tracking | MLflow + MinIO |
| Automation | n8n |

## Architecture

See [docs/architecture.md](docs/architecture.md) for full diagrams and component details.

```
  Browser → Next.js :3000 → FastAPI :8000 → Qdrant :6333
                                          → LM Studio :1234
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service health check |
| POST | `/ask` | Basic RAG pipeline |
| POST | `/ask-graph` | LangGraph pipeline with safety layers |
| POST | `/ask/stream` | SSE streaming (basic pipeline) |
| POST | `/ask-graph/stream` | SSE streaming (graph pipeline) |
| POST | `/ingest` | Document ingestion |

## Service URLs

| Service | URL |
|---------|-----|
| Chat UI | http://localhost:3000 |
| API docs | http://localhost:8000/docs |
| Qdrant | http://localhost:6333/dashboard |
| MLflow | http://localhost:5001 |
| MinIO | http://localhost:9001 |
| n8n | http://localhost:5678 |

## Data Ingestion

```bash
# Download SQuAD dataset
python scripts/download_squad.py

# Ingest documents (requires Qdrant running)
python scripts/ingest.py

# Or via API
curl -X POST http://localhost:8000/ingest -H "X-API-Key: $API_SHARED_SECRET"
```

## Common Commands

```bash
make up             # start services
make down           # stop services
make logs-api       # API logs
make test           # health check all services
```

## Documentation

- [Architecture](docs/architecture.md) — system overview, pipeline flow, DI wiring
- [ADR-001: LangGraph Pipeline](docs/decisions/001-langgraph-pipeline.md)
- [ADR-002: FastAPI Lifespan DI](docs/decisions/002-fastapi-lifespan-di.md)
- [ADR-003: Qdrant Vector Store](docs/decisions/003-qdrant-vector-store.md)
- [ADR-004: SSE Streaming](docs/decisions/004-sse-streaming.md)
- [Frontend](frontend/README.md) — Next.js chat UI, SSE streaming hook, components
- [Design System](DESIGN.md) — typography, colors, spacing, components

## License

MIT
