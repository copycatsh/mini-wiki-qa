# ADR-004: SSE Streaming

**Status:** Accepted
**Date:** 2026-04-04

## Context

The chat UI needs to show LLM responses as they generate, not wait for the full answer. Two transport options:

| Transport | Protocol | Direction | Connection |
|-----------|----------|-----------|------------|
| WebSocket | ws:// | Bidirectional | Persistent |
| SSE | HTTP | Server → Client | Request-scoped |

## Decision

Use Server-Sent Events (SSE) via HTTP POST.

Key factors:
1. **Unidirectional** is sufficient. The client sends a question (POST), the server streams the answer. No need for bidirectional communication.
2. **POST support.** The native `EventSource` API only supports GET. We use `fetch()` with `ReadableStream` instead, which supports POST with a JSON body.
3. **Simpler infrastructure.** SSE works over standard HTTP. No WebSocket upgrade, no connection management, no reconnection protocol.

### Event protocol

Each SSE line is `data: {json}\n\n` where the JSON payload has `event` and `data` fields:

| Event | Data type | Description |
|-------|-----------|-------------|
| `token` | string | Streamed text chunk |
| `citations` | Citation[] | Source documents after stream completes |
| `metadata` | object | Pipeline info, chunk count, latency |
| `error` | string | Error message (generic, no internal details) |
| `done` | string (empty) | Stream complete signal |

### Dual pipeline behavior

- **`/ask/stream`**: Real token streaming. The LLM generates tokens via `astream()`, each chunk is sent as a `token` event immediately. The user sees words appear as the LLM thinks.

- **`/ask-graph/stream`**: Simulated streaming. The graph runs to completion via `ainvoke()`, then the finished answer is word-split into `token` events. The user sees a delay (graph execution) followed by rapid text appearance. Our generate node calls the LLM synchronously via `generator.generate()`. Enabling real token streaming through the graph would require refactoring to use LangGraph's `astream_events` API, which was out of scope for the initial implementation.

### Frontend implementation

The `useChat` hook in `hooks/use-chat.ts` manages the streaming lifecycle:
- Creates `AbortController` for user-initiated stop
- Parses SSE via `streamAsk()` async generator in `lib/api.ts`
- Dispatches events to React state (messages, citations, metadata)
- Handles abort, error, and stream-ended-without-done edge cases

## Consequences

**Good:**
- Simple, HTTP-based, works through any proxy/CDN
- `AbortController` provides clean cancellation
- Event protocol is extensible (add new event types without breaking clients)

**Trade-off:**
- No bidirectional communication. If we later need server-initiated messages (e.g., "new documents indexed"), we'd need to add a separate mechanism.
- Graph pipeline streaming is simulated. A future improvement would be to refactor graph nodes to support token-level streaming, but this requires changes to LangGraph's execution model.
