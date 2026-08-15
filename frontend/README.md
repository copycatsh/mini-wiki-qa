# Rag Playground — Frontend

Chat interface for the Rag Playground. Built with Next.js 14, React 18, shadcn/ui, and Tailwind CSS.

## Quick Start

```bash
# Install dependencies
npm install

# Start dev server (requires backend running on :8000)
npm run dev
```

Open http://localhost:3000.

## Architecture

```
src/
├── app/              # Next.js app router (layout, page, providers)
├── components/
│   ├── chat/         # ChatMessage, ChatInput, CitationList, EmptyState, MetadataTags
│   ├── sidebar/      # Sidebar (pipeline, rerank, multi-query toggles), ThemeToggle, MobileSettings
│   └── ui/           # shadcn/ui primitives (button, input, select, sheet, etc.)
├── hooks/
│   └── use-chat.ts   # Core hook: SSE streaming, message state, conversation history, abort control
├── lib/
│   ├── api.ts        # API client with streamAsk() async generator
│   └── utils.ts      # Tailwind merge utility
└── types/
    └── chat.ts       # ChatMessage, Citation, AskRequest, HistoryMessage, SSEEvent, Pipeline types
```

## SSE Streaming

The `useChat` hook connects to `/ask/stream` or `/ask-graph/stream` via `fetch` + `ReadableStream`. Each request includes the last 5 conversation pairs as history for context-aware answers. Events flow as:

1. `token` — streamed text chunks appended to the message
2. `citations` — source documents with relevance scores
3. `metadata` — pipeline info, chunk count, latency
4. `done` — stream complete
5. `error` — backend error (displayed inline)

Users can abort mid-stream via the Stop button.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend API base URL |
| `NEXT_PUBLIC_API_KEY` | `change-me-in-production` | API key (localhost-only, see api.ts) |

## Docker

The frontend runs as a service in `compose.yml`:

```bash
docker compose up frontend
```

Exposed on port 3000. Connects to the `api` service internally.
