# TODOS

## Error handling at RAG integration boundaries
**What:** Add targeted try/except in retrieval.py, generation.py, ingest.py at external service boundaries (Qdrant, LLM, embedding model).
**Why:** All failures currently bubble up as raw exceptions to a single broad `except Exception` in the API layer, producing generic "Internal error: {str(e)}" 500 responses. Users can't distinguish transient errors (retry-worthy) from permanent misconfigurations.
**Pros:** Clear error messages per failure type. Distinguish Qdrant-down vs. LLM-timeout vs. wrong-embedding-model. Users can self-diagnose.
**Cons:** ~30min of work. Need to decide error taxonomy (custom exceptions vs. standard).
**Context:** Identified during eng review of DI refactoring (2026-04-04). The DI refactoring makes this easier — services are injected, so wrapping calls is straightforward. Start with retrieval.py (most common failure: Qdrant unreachable) and generation.py (LLM timeout/auth errors). Note: streaming endpoints (`ask_stream`, `ask_graph_stream` in `rag_service.py`) already have proper error handling with server-side logging and generic client messages (added in feature/frontend-chat-ui PR). This TODO covers the sync paths only.
**Depends on:** DI refactoring (feature/refactor-di-service-layer) — completed 2026-04-04.

## Async ingestion with BackgroundTasks
**What:** Convert POST /ingest from synchronous to async using FastAPI BackgroundTasks. Return 202 Accepted immediately with a task ID. Add GET /ingest/status/{task_id} for polling.
**Why:** run_ingestion() loads files, chunks, embeds, and indexes. On large document sets this takes minutes, hitting gateway timeouts (typically 30-60s). Currently the endpoint blocks until complete.
**Pros:** No gateway timeouts. Users get immediate feedback. Can poll for progress. Enables ingesting large doc sets.
**Cons:** Need to track task state somewhere (in-memory dict is simplest, Redis for persistence). Adds complexity to admin router.
**Context:** Identified by outside voice during eng review (2026-04-04). For the current small SQuAD dataset, sync is fine. This becomes a problem when ingesting larger document collections.
**Depends on:** /ingest wired to run_ingestion() — completed 2026-04-04 (feature/refactor-di-service-layer).

## Dynamic token budget for conversation history
**What:** Count tokens for system prompt + conversation history + retrieved chunks, dynamically truncate history to fit model's context window.
**Why:** Currently using a fixed MAX_HISTORY_PAIRS=5 heuristic. This works for phi-3-mini 4k but could overflow with verbose conversations where individual messages are long.
**Pros:** Guarantees history never overflows context window. Adapts automatically to different model context sizes.
**Cons:** Requires adding a tokenizer dependency (tiktoken or model-specific). Adds complexity to message construction path.
**Context:** Identified by outside voice during eng review (2026-04-04). The 5-pair limit is a reasonable heuristic for development. This becomes important when supporting models with different context window sizes or when conversations have very long messages.
**Depends on:** Conversational memory (feature/conversational-memory) — in progress.
