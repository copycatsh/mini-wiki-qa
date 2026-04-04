# Frontend Chat UI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Claude.ai-inspired chat frontend for Mini-Wiki Q&A with SSE streaming, citations, and settings sidebar.

**Architecture:** Two workstreams. Lane 1: Add SSE streaming endpoints to FastAPI backend (`async def` + LangChain `.astream()`). Lane 2: New Next.js 14 App Router frontend in `frontend/` with shadcn/ui, Tailwind, TanStack Query. Lane 3: Integration (Docker Compose, env config). All design decisions in `DESIGN.md`.

**Tech Stack:** FastAPI (SSE streaming), Next.js 14 (App Router), shadcn/ui, Tailwind CSS, TanStack Query, Geist + Instrument Serif + Source Sans 3 fonts.

**Key files to read before starting:**
- `DESIGN.md` — all visual/interaction specs (colors, fonts, spacing, states, responsive, a11y)
- `app/api/routers/ask.py` — existing sync endpoints to mirror for streaming
- `app/services/rag_service.py` — service layer to extend with streaming methods
- `app/rag/generation.py` — `AnswerGenerator` with `ChatOpenAI` (has `.astream()`)
- `app/rag/graph.py` — LangGraph with `create_rag_graph()` (compiled graph has `.astream()`)
- `app/tests/conftest.py` — existing test fixtures (mock_retriever, mock_generator, etc.)
- `app/api/dependencies.py` — DI functions for injecting components
- `app/api/schemas.py` — existing Pydantic schemas (AskRequest, AskResponse, Citation)

---

## Lane 1: Backend SSE Streaming

### Task 1: SSE Streaming Schema

**Files:**
- Modify: `app/api/schemas.py`

**Step 1: Add SSE event schemas**

Add these models to `app/api/schemas.py` after the existing `AskResponse`:

```python
class StreamEvent(BaseModel):
    """SSE event sent during streaming"""
    event: str = Field(..., description="Event type: token, citations, metadata, error, done")
    data: str = Field(..., description="Event payload")


class StreamTokenData(BaseModel):
    """Payload for 'token' events"""
    token: str


class StreamCitationsData(BaseModel):
    """Payload for 'citations' event, sent after stream completes"""
    citations: List[Citation]


class StreamMetadataData(BaseModel):
    """Payload for 'metadata' event, sent after stream completes"""
    metadata: dict


class StreamErrorData(BaseModel):
    """Payload for 'error' events"""
    message: str
    code: str = "internal_error"
```

**Step 2: Commit**

```bash
git add app/api/schemas.py
git commit -m "feat(api): add SSE streaming event schemas"
```

---

### Task 2: RAGService Streaming Methods

**Files:**
- Modify: `app/services/rag_service.py`
- Test: `app/tests/test_streaming_service.py`

**Step 1: Write the failing test**

Create `app/tests/test_streaming_service.py`:

```python
"""Tests for RAGService streaming methods"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from services.rag_service import RAGService


@pytest.fixture
def mock_retriever():
    m = MagicMock()
    m.retrieve.return_value = [
        {"text": "Test chunk content about RAG", "source": "test/doc.md", "score": 0.95}
    ]
    return m


@pytest.fixture
def mock_generator():
    m = MagicMock()
    m.llm = MagicMock()
    m.prompt = MagicMock()
    return m


@pytest.fixture
def mock_reranker():
    m = MagicMock()
    m.rerank.return_value = [
        {"text": "Test chunk content about RAG", "source": "test/doc.md", "score": 0.95, "rerank_score": 0.99}
    ]
    return m


@pytest.fixture
def mock_injection_guard():
    m = MagicMock()
    m.check.return_value = {"is_safe": True, "detected_patterns": [], "risk_level": "none"}
    return m


@pytest.fixture
def service(mock_retriever, mock_generator, mock_reranker, mock_injection_guard):
    return RAGService(mock_retriever, mock_generator, mock_reranker)


@pytest.mark.asyncio
async def test_ask_stream_yields_tokens(service, mock_generator):
    """Streaming should yield token events then citations"""
    # Mock the async stream from LangChain
    async def mock_astream(*args, **kwargs):
        for token in ["Hello", " world", "!"]:
            chunk = MagicMock()
            chunk.content = token
            yield chunk

    mock_generator.llm.astream = mock_astream
    mock_generator.prompt.format_messages.return_value = [{"role": "user", "content": "test"}]

    events = []
    async for event_type, data in service.ask_stream("What is RAG?"):
        events.append((event_type, data))

    token_events = [(t, d) for t, d in events if t == "token"]
    assert len(token_events) == 3
    assert token_events[0][1] == "Hello"
    assert token_events[1][1] == " world"
    assert token_events[2][1] == "!"

    citation_events = [(t, d) for t, d in events if t == "citations"]
    assert len(citation_events) == 1

    done_events = [(t, d) for t, d in events if t == "done"]
    assert len(done_events) == 1


@pytest.mark.asyncio
async def test_ask_stream_with_rerank(service, mock_generator, mock_reranker):
    """Streaming with rerank should call reranker before generating"""
    async def mock_astream(*args, **kwargs):
        chunk = MagicMock()
        chunk.content = "Answer"
        yield chunk

    mock_generator.llm.astream = mock_astream
    mock_generator.prompt.format_messages.return_value = [{"role": "user", "content": "test"}]

    events = []
    async for event_type, data in service.ask_stream("What is RAG?", use_rerank=True):
        events.append((event_type, data))

    mock_reranker.rerank.assert_called_once()


@pytest.mark.asyncio
async def test_ask_stream_error_during_generation(service, mock_generator):
    """Stream should yield error event if LLM fails mid-generation"""
    async def mock_astream_error(*args, **kwargs):
        yield MagicMock(content="Partial")
        raise RuntimeError("LLM connection lost")

    mock_generator.llm.astream = mock_astream_error
    mock_generator.prompt.format_messages.return_value = [{"role": "user", "content": "test"}]

    events = []
    async for event_type, data in service.ask_stream("What is RAG?"):
        events.append((event_type, data))

    error_events = [(t, d) for t, d in events if t == "error"]
    assert len(error_events) == 1
    assert "LLM connection lost" in str(error_events[0][1])
```

**Step 2: Run test to verify it fails**

Run: `cd app && python -m pytest tests/test_streaming_service.py -v`
Expected: FAIL with `AttributeError: 'RAGService' object has no attribute 'ask_stream'`

**Step 3: Install pytest-asyncio if not present**

Run: `cd app && pip install pytest-asyncio`

**Step 4: Implement streaming methods**

Add to `app/services/rag_service.py`:

```python
import asyncio
from typing import AsyncGenerator, Tuple, Any

# Add these methods to the RAGService class:

    async def ask_stream(
        self, query: str, top_k: int = 5, use_rerank: bool = False
    ) -> AsyncGenerator[Tuple[str, Any], None]:
        """
        Stream RAG response token by token via async generator.

        Yields tuples of (event_type, data):
        - ("token", "text") for each token
        - ("citations", [...]) after generation completes
        - ("metadata", {...}) after generation completes
        - ("error", "message") on failure
        - ("done", "") when finished
        """
        logger.info(f"[Stream] Processing query: {query[:50]}...")

        try:
            # Sync retrieval in threadpool (Qdrant is sync)
            chunks = await asyncio.to_thread(
                self.retriever.retrieve, query, top_k=top_k
            )

            if use_rerank:
                logger.info("[Stream] Applying reranking...")
                chunks = await asyncio.to_thread(
                    self.reranker.rerank, query, chunks, top_k=top_k
                )

            # Format context
            context = "\n\n---\n\n".join([
                f"Document: {chunk['source']}\n{chunk['text']}"
                for chunk in chunks
            ])

            messages = self.generator.prompt.format_messages(
                context=context, question=query
            )

            # Stream tokens from LLM
            full_answer = ""
            async for chunk in self.generator.llm.astream(messages):
                token = chunk.content
                if token:
                    full_answer += token
                    yield ("token", token)

            # Build citations
            citations = [
                {
                    "document": chunk["source"].split("/")[-1],
                    "chunk_id": f"chunk_{idx}",
                    "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                    "score": chunk.get("rerank_score", chunk.get("score", 0.0)),
                }
                for idx, chunk in enumerate(chunks)
            ]

            yield ("citations", citations)
            yield ("metadata", {
                "query": query,
                "top_k": top_k,
                "use_rerank": use_rerank,
                "chunks_retrieved": len(chunks),
            })
            yield ("done", "")

        except Exception as e:
            logger.error(f"[Stream] Error: {str(e)}", exc_info=True)
            yield ("error", str(e))

    async def ask_graph_stream(
        self, query: str, use_rerank: bool = False, graph=None
    ) -> AsyncGenerator[Tuple[str, Any], None]:
        """
        Stream RAG response via LangGraph pipeline.
        Runs injection_guard → retrieve → rerank → generate(stream) → skip PII for streaming.
        """
        if graph is None:
            yield ("error", "RAG graph is not initialized")
            return

        logger.info(f"[GraphStream] Processing query: {query[:50]}...")

        try:
            # Run pre-generation steps synchronously via graph
            # Then stream the generation step
            initial_state = {
                "query": query,
                "chunks": [],
                "answer": "",
                "use_rerank": use_rerank,
                "metadata": {},
                "is_safe": True,
                "error": "",
            }

            # Use graph.astream to get intermediate states
            final_state = None
            async for state in graph.astream(initial_state):
                final_state = state

            if final_state is None:
                yield ("error", "Graph returned no state")
                return

            # Check the final state from the last node
            last_node = list(final_state.keys())[-1]
            state_data = final_state[last_node]

            if not state_data.get("is_safe", True):
                yield ("error", state_data.get("error", "Query blocked by safety filter"))
                return

            answer = state_data.get("answer", "")
            chunks = state_data.get("chunks", [])

            # Stream the answer token by token (simulated from complete answer)
            # LangGraph returns complete state, so we chunk the answer for streaming UX
            words = answer.split(" ")
            for i, word in enumerate(words):
                token = word if i == 0 else " " + word
                yield ("token", token)

            citations = [
                {
                    "document": chunk["source"].split("/")[-1],
                    "chunk_id": f"chunk_{idx}",
                    "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                    "score": chunk.get("rerank_score", chunk.get("score", 0.0)),
                }
                for idx, chunk in enumerate(chunks)
            ]

            yield ("citations", citations)
            yield ("metadata", {
                **state_data.get("metadata", {}),
                "query": query,
                "use_rerank": use_rerank,
                "pipeline": "langgraph",
            })
            yield ("done", "")

        except Exception as e:
            logger.error(f"[GraphStream] Error: {str(e)}", exc_info=True)
            yield ("error", str(e))
```

**Step 5: Run tests to verify they pass**

Run: `cd app && python -m pytest tests/test_streaming_service.py -v`
Expected: All 3 tests PASS

**Step 6: Commit**

```bash
git add app/services/rag_service.py app/tests/test_streaming_service.py
git commit -m "feat(api): add SSE streaming methods to RAGService"
```

---

### Task 3: SSE Streaming Router Endpoints

**Files:**
- Create: `app/api/routers/stream.py`
- Modify: `app/api/main.py` (add router import)
- Test: `app/tests/test_stream.py`

**Step 1: Write the failing test**

Create `app/tests/test_stream.py`:

```python
"""Tests for /ask/stream and /ask-graph/stream SSE endpoints"""
import json
import pytest
from unittest.mock import MagicMock, AsyncMock
from httpx import AsyncClient, ASGITransport
from tests.conftest import VALID_API_KEY
from api.main import app
from api.dependencies import (
    get_retriever, get_generator, get_reranker, get_rag_graph, verify_api_key,
)


@pytest.fixture
def mock_retriever():
    m = MagicMock()
    m.retrieve.return_value = [
        {"text": "Test chunk content", "source": "test/doc.md", "score": 0.95}
    ]
    return m


@pytest.fixture
def mock_generator():
    m = MagicMock()
    m.prompt = MagicMock()
    m.prompt.format_messages.return_value = [{"role": "user", "content": "test"}]

    async def mock_astream(*args, **kwargs):
        for token in ["Hello", " world"]:
            chunk = MagicMock()
            chunk.content = token
            yield chunk

    m.llm = MagicMock()
    m.llm.astream = mock_astream
    return m


@pytest.fixture
def mock_reranker():
    m = MagicMock()
    m.rerank.return_value = [
        {"text": "Test chunk", "source": "test/doc.md", "score": 0.95, "rerank_score": 0.99}
    ]
    return m


@pytest.fixture
def mock_rag_graph():
    m = MagicMock()

    async def mock_astream(state):
        yield {
            "pii_scrubber": {
                "query": state["query"],
                "chunks": [{"text": "Test chunk", "source": "test/doc.md", "score": 0.9}],
                "answer": "Graph streamed answer",
                "use_rerank": False,
                "metadata": {"injection_check": {"is_safe": True}},
                "is_safe": True,
                "error": "",
            }
        }

    m.astream = mock_astream
    return m


@pytest.fixture
def setup_overrides(mock_retriever, mock_generator, mock_reranker, mock_rag_graph):
    app.dependency_overrides[get_retriever] = lambda: mock_retriever
    app.dependency_overrides[get_generator] = lambda: mock_generator
    app.dependency_overrides[get_reranker] = lambda: mock_reranker
    app.dependency_overrides[get_rag_graph] = lambda: mock_rag_graph
    app.dependency_overrides[verify_api_key] = lambda: VALID_API_KEY
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_ask_stream_happy_path(setup_overrides):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/ask/stream",
            json={"query": "What is RAG?"},
            headers={"X-API-Key": VALID_API_KEY},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        lines = resp.text.strip().split("\n")
        events = []
        for line in lines:
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

        token_events = [e for e in events if e["event"] == "token"]
        assert len(token_events) >= 1

        done_events = [e for e in events if e["event"] == "done"]
        assert len(done_events) == 1


@pytest.mark.asyncio
async def test_ask_stream_requires_api_key(setup_overrides):
    # Remove the API key override so the real dependency runs
    app.dependency_overrides.pop(verify_api_key, None)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/ask/stream",
            json={"query": "What is RAG?"},
        )
        assert resp.status_code == 422


@pytest.mark.asyncio
async def test_ask_graph_stream_happy_path(setup_overrides):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/ask-graph/stream",
            json={"query": "What is RAG?"},
            headers={"X-API-Key": VALID_API_KEY},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        lines = resp.text.strip().split("\n")
        events = []
        for line in lines:
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

        done_events = [e for e in events if e["event"] == "done"]
        assert len(done_events) == 1


@pytest.mark.asyncio
async def test_ask_stream_empty_query(setup_overrides):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/ask/stream",
            json={"query": ""},
            headers={"X-API-Key": VALID_API_KEY},
        )
        assert resp.status_code == 422
```

**Step 2: Run tests to verify they fail**

Run: `cd app && python -m pytest tests/test_stream.py -v`
Expected: FAIL (no `/ask/stream` route)

**Step 3: Create the streaming router**

Create `app/api/routers/stream.py`:

```python
"""SSE streaming endpoints for RAG Q&A"""
import json
import logging
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from api.schemas import AskRequest
from api.dependencies import (
    verify_api_key, get_retriever, get_generator, get_reranker, get_rag_graph,
)
from services.rag_service import RAGService

logger = logging.getLogger(__name__)
router = APIRouter()


async def sse_generator(stream):
    """Convert async generator of (event_type, data) tuples to SSE format"""
    try:
        async for event_type, data in stream:
            payload = json.dumps({"event": event_type, "data": data})
            yield f"data: {payload}\n\n"
    except Exception as e:
        logger.error(f"SSE stream error: {e}", exc_info=True)
        payload = json.dumps({"event": "error", "data": str(e)})
        yield f"data: {payload}\n\n"


@router.post("/ask/stream", tags=["RAG Streaming"])
async def ask_stream(
    request: AskRequest,
    api_key: str = Depends(verify_api_key),
    retriever=Depends(get_retriever),
    generator=Depends(get_generator),
    reranker=Depends(get_reranker),
):
    service = RAGService(retriever, generator, reranker)
    stream = service.ask_stream(
        request.query, top_k=request.top_k, use_rerank=request.use_rerank
    )
    return StreamingResponse(
        sse_generator(stream),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/ask-graph/stream", tags=["RAG Streaming"])
async def ask_graph_stream(
    request: AskRequest,
    api_key: str = Depends(verify_api_key),
    retriever=Depends(get_retriever),
    generator=Depends(get_generator),
    reranker=Depends(get_reranker),
    graph=Depends(get_rag_graph),
):
    service = RAGService(retriever, generator, reranker)
    stream = service.ask_graph_stream(
        request.query, use_rerank=request.use_rerank, graph=graph
    )
    return StreamingResponse(
        sse_generator(stream),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
```

**Step 4: Register the router in main.py**

Add to `app/api/main.py` after the existing router imports (line 65):

```python
from api.routers.stream import router as stream_router
```

And after the existing `app.include_router` calls (line 69):

```python
app.include_router(stream_router)
```

**Step 5: Run tests to verify they pass**

Run: `cd app && python -m pytest tests/test_stream.py -v`
Expected: All 4 tests PASS

**Step 6: Run ALL existing tests to check for regressions**

Run: `cd app && python -m pytest tests/ -v`
Expected: All tests PASS (existing + new)

**Step 7: Commit**

```bash
git add app/api/routers/stream.py app/api/main.py app/tests/test_stream.py
git commit -m "feat(api): add SSE streaming endpoints /ask/stream and /ask-graph/stream"
```

---

## Lane 2: Next.js Frontend

### Task 4: Initialize Next.js Project

**Files:**
- Create: `frontend/` directory with Next.js 14 App Router

**Step 1: Create the Next.js project**

```bash
cd /Users/anton/WorkProjects/pet_projects/mini-wiki-qa
npx create-next-app@14 frontend --typescript --tailwind --eslint --app --src-dir --import-alias "@/*" --no-turbo
```

When prompted: use defaults (Yes to all).

**Step 2: Install dependencies**

```bash
cd frontend
npm install @tanstack/react-query geist
npm install -D @types/node
```

**Step 3: Initialize shadcn/ui**

```bash
cd frontend
npx shadcn@latest init
```

When prompted:
- Style: Default
- Base color: Neutral
- CSS variables: Yes

**Step 4: Install shadcn components we need**

```bash
cd frontend
npx shadcn@latest add button input select toggle accordion alert sheet
```

**Step 5: Verify it builds**

Run: `cd frontend && npm run build`
Expected: Build succeeds

**Step 6: Commit**

```bash
git add frontend/
git commit -m "feat(frontend): initialize Next.js 14 with shadcn/ui, Tailwind, TanStack Query"
```

---

### Task 5: Configure Design System Theme

**Files:**
- Modify: `frontend/tailwind.config.ts`
- Modify: `frontend/src/app/globals.css`
- Modify: `frontend/src/app/layout.tsx`

All values from `DESIGN.md`. Read it before implementing.

**Step 1: Configure Tailwind with custom theme**

Replace `frontend/tailwind.config.ts`:

```typescript
import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-geist-sans)", "system-ui", "sans-serif"],
        serif: ["var(--font-instrument-serif)", "Georgia", "serif"],
        body: ["var(--font-source-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-geist-mono)", "Consolas", "monospace"],
      },
      colors: {
        background: "var(--background)",
        surface: "var(--surface)",
        sidebar: "var(--sidebar)",
        "code-bg": "var(--code-bg)",
        border: "var(--border)",
        text: {
          DEFAULT: "var(--text)",
          muted: "var(--text-muted)",
        },
        accent: {
          DEFAULT: "var(--accent)",
          hover: "var(--accent-hover)",
        },
        success: "var(--success)",
        warning: "var(--warning)",
        error: "var(--error)",
        info: "var(--info)",
      },
      maxWidth: {
        chat: "720px",
      },
      width: {
        sidebar: "240px",
      },
      borderRadius: {
        sm: "4px",
        md: "8px",
        lg: "12px",
      },
      spacing: {
        "2xs": "2px",
        xs: "4px",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};
export default config;
```

**Step 2: Set CSS custom properties in globals.css**

Replace `frontend/src/app/globals.css`:

```css
@import "tailwindcss";
@import "tw-animate-css";

@custom-variant dark (&:is(.dark *));

@theme inline {
  --color-background: var(--background);
  --color-foreground: var(--text);
  --font-sans: var(--font-geist-sans), system-ui, sans-serif;
  --font-mono: var(--font-geist-mono), Consolas, monospace;
}

:root {
  --background: #FAF9F6;
  --surface: #FFFFFF;
  --sidebar: #F5F3EF;
  --code-bg: #F5F3EF;
  --border: #E8E5E0;
  --text: #1A1A1A;
  --text-muted: #6B6B6B;
  --accent: #C2713A;
  --accent-hover: #A85E30;
  --success: #2D8A4E;
  --warning: #B8860B;
  --error: #C53030;
  --info: #2B6CB0;
}

.dark {
  --background: #1A1917;
  --surface: #242320;
  --sidebar: #1F1E1B;
  --code-bg: #2A2927;
  --border: #3A3835;
  --text: #E8E5E0;
  --text-muted: #9A9890;
  --accent: #D4845A;
  --accent-hover: #E09468;
}

body {
  font-family: var(--font-geist-sans), system-ui, sans-serif;
  background: var(--background);
  color: var(--text);
  font-size: 14px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}
```

**Step 3: Configure fonts in layout.tsx**

Replace `frontend/src/app/layout.tsx`:

```typescript
import type { Metadata } from "next";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";
import { Source_Sans_3, Instrument_Serif } from "next/font/google";
import "./globals.css";
import { Providers } from "./providers";

const sourceSans = Source_Sans_3({
  subsets: ["latin"],
  variable: "--font-source-sans",
  display: "swap",
});

const instrumentSerif = Instrument_Serif({
  weight: "400",
  subsets: ["latin"],
  variable: "--font-instrument-serif",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Mini Wiki Q&A",
  description: "Ask your documents anything",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      className={`${GeistSans.variable} ${GeistMono.variable} ${sourceSans.variable} ${instrumentSerif.variable}`}
      suppressHydrationWarning
    >
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
```

**Step 4: Create providers (TanStack Query)**

Create `frontend/src/app/providers.tsx`:

```typescript
"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";

export function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(() => new QueryClient());

  return (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
}
```

**Step 5: Verify it builds**

Run: `cd frontend && npm run build`
Expected: Build succeeds

**Step 6: Commit**

```bash
git add frontend/
git commit -m "feat(frontend): configure design system theme from DESIGN.md"
```

---

### Task 6: API Client & SSE Hook

**Files:**
- Create: `frontend/src/lib/api.ts`
- Create: `frontend/src/hooks/use-chat.ts`
- Create: `frontend/src/types/chat.ts`

**Step 1: Define types**

Create `frontend/src/types/chat.ts`:

```typescript
export interface Citation {
  document: string;
  chunk_id: string;
  text: string;
  score: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  metadata?: Record<string, unknown>;
  isStreaming?: boolean;
  error?: string;
}

export interface AskRequest {
  query: string;
  top_k?: number;
  use_rerank?: boolean;
}

export type Pipeline = "/ask" | "/ask-graph";

export interface SSEEvent {
  event: "token" | "citations" | "metadata" | "error" | "done";
  data: unknown;
}
```

**Step 2: Create API client with SSE support**

Create `frontend/src/lib/api.ts`:

```typescript
import { SSEEvent } from "@/types/chat";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "change-me-in-production";

export async function fetchHealth() {
  const resp = await fetch(`${API_URL}/health`, {
    headers: { "X-API-Key": API_KEY },
  });
  if (!resp.ok) throw new Error(`Health check failed: ${resp.status}`);
  return resp.json();
}

export async function* streamAsk(
  query: string,
  pipeline: "/ask" | "/ask-graph",
  options: { use_rerank?: boolean; top_k?: number } = {},
  signal?: AbortSignal
): AsyncGenerator<SSEEvent> {
  const resp = await fetch(`${API_URL}${pipeline}/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": API_KEY,
    },
    body: JSON.stringify({
      query,
      use_rerank: options.use_rerank ?? false,
      top_k: options.top_k ?? 5,
    }),
    signal,
  });

  if (!resp.ok) {
    throw new Error(`API error: ${resp.status} ${resp.statusText}`);
  }

  const reader = resp.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          const event: SSEEvent = JSON.parse(line.slice(6));
          yield event;
        } catch {
          // Skip malformed events
        }
      }
    }
  }
}
```

**Step 3: Create the chat hook**

Create `frontend/src/hooks/use-chat.ts`:

```typescript
"use client";

import { useState, useCallback, useRef } from "react";
import { ChatMessage, Citation, Pipeline } from "@/types/chat";
import { streamAsk } from "@/lib/api";

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const sendMessage = useCallback(
    async (query: string, pipeline: Pipeline, useRerank: boolean) => {
      const userMessage: ChatMessage = {
        id: `user-${Date.now()}`,
        role: "user",
        content: query,
      };

      const assistantId = `assistant-${Date.now()}`;
      const assistantMessage: ChatMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        isStreaming: true,
      };

      setMessages((prev) => [...prev, userMessage, assistantMessage]);
      setIsStreaming(true);

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const stream = streamAsk(query, pipeline, { use_rerank: useRerank }, controller.signal);

        for await (const event of stream) {
          switch (event.event) {
            case "token":
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? { ...m, content: m.content + (event.data as string) }
                    : m
                )
              );
              break;

            case "citations":
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? { ...m, citations: event.data as Citation[] }
                    : m
                )
              );
              break;

            case "metadata":
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? { ...m, metadata: event.data as Record<string, unknown> }
                    : m
                )
              );
              break;

            case "error":
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? { ...m, error: event.data as string, isStreaming: false }
                    : m
                )
              );
              break;

            case "done":
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId ? { ...m, isStreaming: false } : m
                )
              );
              break;
          }
        }
      } catch (err) {
        if ((err as Error).name !== "AbortError") {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? {
                    ...m,
                    error: (err as Error).message || "Something went wrong",
                    isStreaming: false,
                  }
                : m
            )
          );
        }
      } finally {
        setIsStreaming(false);
        abortRef.current = null;
      }
    },
    []
  );

  const stopStreaming = useCallback(() => {
    abortRef.current?.abort();
    setIsStreaming(false);
    setMessages((prev) =>
      prev.map((m) => (m.isStreaming ? { ...m, isStreaming: false } : m))
    );
  }, []);

  const clearMessages = useCallback(() => {
    setMessages([]);
  }, []);

  return { messages, isStreaming, sendMessage, stopStreaming, clearMessages };
}
```

**Step 4: Verify it builds**

Run: `cd frontend && npm run build`
Expected: Build succeeds

**Step 5: Commit**

```bash
git add frontend/src/types/ frontend/src/lib/ frontend/src/hooks/
git commit -m "feat(frontend): add API client, SSE streaming, and useChat hook"
```

---

### Task 7: Chat UI Components

**Files:**
- Create: `frontend/src/components/chat/chat-message.tsx`
- Create: `frontend/src/components/chat/citation-list.tsx`
- Create: `frontend/src/components/chat/metadata-tags.tsx`
- Create: `frontend/src/components/chat/chat-input.tsx`
- Create: `frontend/src/components/chat/empty-state.tsx`

All visual specs from `DESIGN.md` sections: Component Patterns, Interaction States, Information Architecture.

**Step 1: Create ChatMessage component**

Create `frontend/src/components/chat/chat-message.tsx`:

```tsx
"use client";

import { ChatMessage as ChatMessageType } from "@/types/chat";
import { CitationList } from "./citation-list";
import { MetadataTags } from "./metadata-tags";

interface ChatMessageProps {
  message: ChatMessageType;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user";

  return (
    <div className="w-full max-w-chat mx-auto">
      <div className="text-[11px] font-semibold uppercase tracking-[0.08em] text-text-muted mb-1.5">
        {isUser ? "You" : "Mini Wiki"}
      </div>

      {isUser ? (
        <div className="font-sans text-sm font-medium text-text">
          {message.content}
        </div>
      ) : (
        <div>
          <div className="font-body text-[15px] leading-[1.7] text-text">
            {message.content}
            {message.isStreaming && (
              <span className="inline-block w-0.5 h-4 bg-accent animate-pulse ml-0.5 align-text-bottom" />
            )}
          </div>

          {message.error && (
            <div className="mt-2 px-3 py-2 rounded-md border-l-[3px] border-error bg-[#fef2f2] text-sm text-[#9b2c2c] dark:bg-[#2d0f0f] dark:text-[#fc8181]">
              {message.error}
            </div>
          )}

          {message.citations && message.citations.length > 0 && !message.isStreaming && (
            <CitationList citations={message.citations} />
          )}

          {message.metadata && !message.isStreaming && (
            <MetadataTags metadata={message.metadata} />
          )}
        </div>
      )}
    </div>
  );
}
```

**Step 2: Create CitationList component**

Create `frontend/src/components/chat/citation-list.tsx`:

```tsx
"use client";

import { useState } from "react";
import { Citation } from "@/types/chat";

interface CitationListProps {
  citations: Citation[];
}

export function CitationList({ citations }: CitationListProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [expandedChunks, setExpandedChunks] = useState<Set<number>>(new Set());

  const toggleChunk = (idx: number) => {
    setExpandedChunks((prev) => {
      const next = new Set(prev);
      next.has(idx) ? next.delete(idx) : next.add(idx);
      return next;
    });
  };

  return (
    <div className="mt-2.5">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-1.5 px-2.5 py-1.5 bg-code-bg rounded-sm text-xs font-sans text-text-muted hover:text-accent transition-colors duration-150"
        aria-expanded={isOpen}
      >
        <span
          className="text-[10px] transition-transform duration-150"
          style={{ transform: isOpen ? "rotate(90deg)" : "rotate(0deg)" }}
        >
          &#9654;
        </span>
        {citations.length} source{citations.length !== 1 ? "s" : ""}
      </button>

      {isOpen && (
        <div className="mt-2 p-2.5 bg-surface border border-border rounded-sm">
          {citations.map((citation, idx) => (
            <div key={idx} className="py-1 flex gap-2 items-baseline">
              <span className="font-mono text-[10px] bg-code-bg px-1.5 py-0.5 rounded-sm text-accent font-medium shrink-0">
                {idx + 1}
              </span>
              <div className="min-w-0">
                <span className="text-xs font-sans text-accent">
                  {citation.document}
                </span>
                <span className="text-[11px] font-mono text-text-muted ml-2">
                  {(citation.score * 100).toFixed(0)}%
                </span>
                <p className="text-[13px] font-body text-text-muted mt-0.5 leading-relaxed">
                  {expandedChunks.has(idx)
                    ? citation.text
                    : citation.text.length > 200
                      ? citation.text.slice(0, 200) + "..."
                      : citation.text}
                  {citation.text.length > 200 && (
                    <button
                      onClick={() => toggleChunk(idx)}
                      className="ml-1 text-accent text-xs hover:underline"
                    >
                      {expandedChunks.has(idx) ? "Show less" : "Show full text"}
                    </button>
                  )}
                </p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
```

**Step 3: Create MetadataTags component**

Create `frontend/src/components/chat/metadata-tags.tsx`:

```tsx
interface MetadataTagsProps {
  metadata: Record<string, unknown>;
}

export function MetadataTags({ metadata }: MetadataTagsProps) {
  const pipeline = metadata.pipeline as string | undefined;
  const chunksRetrieved = metadata.chunks_retrieved as number | undefined;
  const useRerank = metadata.use_rerank as boolean | undefined;

  return (
    <div className="flex gap-2 mt-2 flex-wrap">
      {chunksRetrieved !== undefined && (
        <span className="font-mono text-[11px] px-2 py-0.5 bg-code-bg text-text-muted rounded-sm">
          {chunksRetrieved} chunks
        </span>
      )}
      {useRerank && (
        <span className="font-mono text-[11px] px-2 py-0.5 bg-[color-mix(in_srgb,var(--success)_12%,transparent)] text-success rounded-sm">
          reranked
        </span>
      )}
      {pipeline && (
        <span className="font-mono text-[11px] px-2 py-0.5 bg-[color-mix(in_srgb,var(--accent)_12%,transparent)] text-accent rounded-sm">
          {pipeline === "langgraph" ? "/ask-graph" : "/ask"}
        </span>
      )}
    </div>
  );
}
```

**Step 4: Create ChatInput component**

Create `frontend/src/components/chat/chat-input.tsx`:

```tsx
"use client";

import { useState, useRef, useCallback, KeyboardEvent } from "react";

interface ChatInputProps {
  onSend: (query: string) => void;
  onStop: () => void;
  isStreaming: boolean;
  disabled?: boolean;
}

export function ChatInput({ onSend, onStop, isStreaming, disabled }: ChatInputProps) {
  const [value, setValue] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSend = useCallback(() => {
    const trimmed = value.trim();
    if (!trimmed || isStreaming) return;
    onSend(trimmed);
    setValue("");
  }, [value, isStreaming, onSend]);

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="border-t border-border bg-background px-8 py-4">
      <div className="max-w-chat mx-auto flex gap-2 items-center">
        <input
          ref={inputRef}
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about your documents..."
          disabled={disabled}
          className="flex-1 font-sans text-sm px-3.5 py-2.5 border border-border rounded-md bg-surface text-text outline-none transition-colors duration-150 focus:border-accent placeholder:text-text-muted disabled:opacity-50"
          aria-label="Ask a question"
        />
        {isStreaming ? (
          <button
            onClick={onStop}
            className="w-10 h-10 rounded-md bg-error flex items-center justify-center shrink-0 transition-colors hover:opacity-90"
            aria-label="Stop streaming"
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="white">
              <rect width="14" height="14" rx="2" />
            </svg>
          </button>
        ) : (
          <button
            onClick={handleSend}
            disabled={!value.trim() || disabled}
            className="w-10 h-10 rounded-md bg-accent flex items-center justify-center shrink-0 transition-colors hover:bg-accent-hover disabled:opacity-50"
            aria-label="Send message"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M22 2L11 13" />
              <path d="M22 2L15 22L11 13L2 9L22 2Z" />
            </svg>
          </button>
        )}
      </div>
    </div>
  );
}
```

**Step 5: Create EmptyState component**

Create `frontend/src/components/chat/empty-state.tsx`:

```tsx
"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchHealth } from "@/lib/api";

interface EmptyStateProps {
  onSuggestionClick: (query: string) => void;
}

const SUGGESTIONS = [
  "What is retrieval-augmented generation?",
  "How does the reranking pipeline work?",
  "What safety mechanisms are in place?",
  "How are documents chunked and indexed?",
];

export function EmptyState({ onSuggestionClick }: EmptyStateProps) {
  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: fetchHealth,
    retry: false,
  });

  return (
    <div className="flex-1 flex items-center justify-center px-8">
      <div className="text-center max-w-md">
        <h1 className="font-serif text-[32px] text-text leading-tight">
          Ask your documents anything
        </h1>

        <div className="mt-6 flex flex-wrap justify-center gap-2">
          {SUGGESTIONS.map((suggestion) => (
            <button
              key={suggestion}
              onClick={() => onSuggestionClick(suggestion)}
              className="px-3 py-1.5 text-xs font-sans border border-border rounded-sm text-text-muted hover:border-accent hover:text-accent transition-colors duration-150"
            >
              {suggestion}
            </button>
          ))}
        </div>

        {health && (
          <p className="mt-4 text-[11px] font-mono text-text-muted">
            {health.status === "healthy" ? "Backend connected" : "Backend degraded"}
          </p>
        )}
      </div>
    </div>
  );
}
```

**Step 6: Verify it builds**

Run: `cd frontend && npm run build`
Expected: Build succeeds

**Step 7: Commit**

```bash
git add frontend/src/components/
git commit -m "feat(frontend): add chat UI components (messages, citations, input, empty state)"
```

---

### Task 8: Sidebar Component

**Files:**
- Create: `frontend/src/components/sidebar/sidebar.tsx`
- Create: `frontend/src/components/sidebar/theme-toggle.tsx`

**Step 1: Create Sidebar component**

Create `frontend/src/components/sidebar/sidebar.tsx`:

```tsx
"use client";

import { Pipeline } from "@/types/chat";
import { ThemeToggle } from "./theme-toggle";

interface SidebarProps {
  pipeline: Pipeline;
  onPipelineChange: (pipeline: Pipeline) => void;
  useRerank: boolean;
  onRerankChange: (useRerank: boolean) => void;
}

export function Sidebar({
  pipeline,
  onPipelineChange,
  useRerank,
  onRerankChange,
}: SidebarProps) {
  return (
    <aside className="w-sidebar min-w-[240px] bg-sidebar border-r border-border px-4 py-5 flex flex-col gap-6 max-lg:hidden">
      <h1 className="font-serif text-xl text-text">Mini Wiki Q&A</h1>

      <div>
        <div className="text-[11px] font-semibold uppercase tracking-[0.08em] text-text-muted mb-2.5">
          Pipeline
        </div>
        <select
          value={pipeline}
          onChange={(e) => onPipelineChange(e.target.value as Pipeline)}
          className="w-full font-sans text-[13px] px-2.5 py-1.5 border border-border rounded-sm bg-surface text-text outline-none appearance-none cursor-pointer focus:border-accent"
          aria-label="Select pipeline"
        >
          <option value="/ask">/ask</option>
          <option value="/ask-graph">/ask-graph</option>
        </select>
      </div>

      <div>
        <div className="text-[11px] font-semibold uppercase tracking-[0.08em] text-text-muted mb-2.5">
          Settings
        </div>
        <div className="flex items-center justify-between py-1.5">
          <span className="font-sans text-[13px] text-text">Reranking</span>
          <button
            role="switch"
            aria-checked={useRerank}
            aria-label="Enable reranking"
            onClick={() => onRerankChange(!useRerank)}
            className={`relative w-10 h-[22px] rounded-full transition-colors duration-200 ${
              useRerank ? "bg-accent" : "bg-border"
            }`}
          >
            <span
              className={`absolute top-0.5 left-0.5 w-[18px] h-[18px] bg-white rounded-full shadow-sm transition-transform duration-200 ${
                useRerank ? "translate-x-[18px]" : ""
              }`}
            />
          </button>
        </div>
      </div>

      <div className="mt-auto">
        <div className="text-[11px] font-semibold uppercase tracking-[0.08em] text-text-muted mb-2.5">
          Info
        </div>
        <div className="font-mono text-[11px] text-text-muted leading-relaxed">
          Pipeline: {pipeline}
          <br />
          Rerank: {useRerank ? "on" : "off"}
        </div>
      </div>

      <ThemeToggle />
    </aside>
  );
}
```

**Step 2: Create ThemeToggle component**

Create `frontend/src/components/sidebar/theme-toggle.tsx`:

```tsx
"use client";

import { useEffect, useState } from "react";

export function ThemeToggle() {
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    setIsDark(document.documentElement.classList.contains("dark"));
  }, []);

  const toggle = () => {
    const next = !isDark;
    setIsDark(next);
    document.documentElement.classList.toggle("dark", next);
  };

  return (
    <button
      onClick={toggle}
      className="font-sans text-xs text-text-muted hover:text-accent transition-colors"
    >
      {isDark ? "Light mode" : "Dark mode"}
    </button>
  );
}
```

**Step 3: Verify it builds**

Run: `cd frontend && npm run build`
Expected: Build succeeds

**Step 4: Commit**

```bash
git add frontend/src/components/sidebar/
git commit -m "feat(frontend): add sidebar with pipeline selector, rerank toggle, theme toggle"
```

---

### Task 9: Main Chat Page

**Files:**
- Modify: `frontend/src/app/page.tsx`

**Step 1: Build the main page**

Replace `frontend/src/app/page.tsx`:

```tsx
"use client";

import { useState, useRef, useEffect } from "react";
import { useChat } from "@/hooks/use-chat";
import { Pipeline } from "@/types/chat";
import { Sidebar } from "@/components/sidebar/sidebar";
import { ChatMessage } from "@/components/chat/chat-message";
import { ChatInput } from "@/components/chat/chat-input";
import { EmptyState } from "@/components/chat/empty-state";

export default function Home() {
  const [pipeline, setPipeline] = useState<Pipeline>("/ask");
  const [useRerank, setUseRerank] = useState(false);
  const { messages, isStreaming, sendMessage, stopStreaming } = useChat();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const handleSend = (query: string) => {
    sendMessage(query, pipeline, useRerank);
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="flex h-screen">
      <Sidebar
        pipeline={pipeline}
        onPipelineChange={setPipeline}
        useRerank={useRerank}
        onRerankChange={setUseRerank}
      />

      <main className="flex-1 flex flex-col min-w-0">
        {messages.length === 0 ? (
          <EmptyState onSuggestionClick={handleSend} />
        ) : (
          <div
            className="flex-1 overflow-y-auto px-8 py-6"
            role="log"
            aria-live="polite"
          >
            <div className="flex flex-col gap-4">
              {messages.map((message) => (
                <ChatMessage key={message.id} message={message} />
              ))}
              <div ref={messagesEndRef} />
            </div>
          </div>
        )}

        <ChatInput
          onSend={handleSend}
          onStop={stopStreaming}
          isStreaming={isStreaming}
        />
      </main>
    </div>
  );
}
```

**Step 2: Verify it builds**

Run: `cd frontend && npm run build`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add frontend/src/app/page.tsx
git commit -m "feat(frontend): assemble main chat page with sidebar, messages, input"
```

---

### Task 10: Mobile Bottom Sheet

**Files:**
- Create: `frontend/src/components/sidebar/mobile-settings.tsx`
- Modify: `frontend/src/app/page.tsx` (add mobile header + sheet)

**Step 1: Create mobile settings sheet**

Create `frontend/src/components/sidebar/mobile-settings.tsx`:

```tsx
"use client";

import { useState } from "react";
import { Pipeline } from "@/types/chat";

interface MobileSettingsProps {
  pipeline: Pipeline;
  onPipelineChange: (pipeline: Pipeline) => void;
  useRerank: boolean;
  onRerankChange: (useRerank: boolean) => void;
}

export function MobileSettings({
  pipeline,
  onPipelineChange,
  useRerank,
  onRerankChange,
}: MobileSettingsProps) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      {/* Mobile header */}
      <div className="lg:hidden flex items-center justify-between px-4 py-3 border-b border-border bg-background">
        <span className="font-serif text-lg text-text">Mini Wiki Q&A</span>
        <button
          onClick={() => setIsOpen(true)}
          className="w-8 h-8 flex items-center justify-center text-text-muted hover:text-accent transition-colors"
          aria-label="Open settings"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="3" />
            <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
          </svg>
        </button>
      </div>

      {/* Backdrop + Sheet */}
      {isOpen && (
        <>
          <div
            className="fixed inset-0 bg-black/30 backdrop-blur-sm z-40"
            onClick={() => setIsOpen(false)}
          />
          <div
            className="fixed bottom-0 left-0 right-0 z-50 bg-surface border-t border-border rounded-t-lg max-h-[280px] p-4"
            role="dialog"
            aria-label="Settings"
          >
            <div className="w-10 h-1 bg-border rounded-full mx-auto mb-4" />

            <div className="mb-4">
              <div className="text-[11px] font-semibold uppercase tracking-[0.08em] text-text-muted mb-2">
                Pipeline
              </div>
              <select
                value={pipeline}
                onChange={(e) => onPipelineChange(e.target.value as Pipeline)}
                className="w-full font-sans text-sm px-3 py-2 border border-border rounded-sm bg-background text-text outline-none"
                aria-label="Select pipeline"
              >
                <option value="/ask">/ask</option>
                <option value="/ask-graph">/ask-graph</option>
              </select>
            </div>

            <div className="flex items-center justify-between py-2">
              <span className="font-sans text-sm text-text">Reranking</span>
              <button
                role="switch"
                aria-checked={useRerank}
                aria-label="Enable reranking"
                onClick={() => onRerankChange(!useRerank)}
                className={`relative w-10 h-[22px] rounded-full transition-colors duration-200 ${
                  useRerank ? "bg-accent" : "bg-border"
                }`}
              >
                <span
                  className={`absolute top-0.5 left-0.5 w-[18px] h-[18px] bg-white rounded-full shadow-sm transition-transform duration-200 ${
                    useRerank ? "translate-x-[18px]" : ""
                  }`}
                />
              </button>
            </div>
          </div>
        </>
      )}
    </>
  );
}
```

**Step 2: Update page.tsx to include mobile settings**

In `frontend/src/app/page.tsx`, add the import:

```typescript
import { MobileSettings } from "@/components/sidebar/mobile-settings";
```

And add `<MobileSettings ... />` inside `<main>` before the chat area:

```tsx
<main className="flex-1 flex flex-col min-w-0">
  <MobileSettings
    pipeline={pipeline}
    onPipelineChange={setPipeline}
    useRerank={useRerank}
    onRerankChange={setUseRerank}
  />
  {/* ... rest of chat area */}
</main>
```

**Step 3: Verify it builds**

Run: `cd frontend && npm run build`
Expected: Build succeeds

**Step 4: Commit**

```bash
git add frontend/src/components/sidebar/mobile-settings.tsx frontend/src/app/page.tsx
git commit -m "feat(frontend): add mobile bottom sheet for settings"
```

---

## Lane 3: Integration

### Task 11: Environment Configuration

**Files:**
- Create: `frontend/.env.local`
- Modify: `compose.yml` (add frontend service)

**Step 1: Create frontend env file**

Create `frontend/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_API_KEY=change-me-in-production
```

**Step 2: Create frontend Dockerfile**

Create `frontend/Dockerfile`:

```dockerfile
FROM node:20-alpine AS base

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

FROM base AS dev
CMD ["npm", "run", "dev"]

FROM base AS build
RUN npm run build

FROM node:20-alpine AS prod
WORKDIR /app
COPY --from=build /app/.next ./.next
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/package.json ./
COPY --from=build /app/public ./public
CMD ["npm", "start"]
```

**Step 3: Add frontend service to compose.yml**

Add this service block to `compose.yml` after the `api` service:

```yaml
  # Next.js Frontend
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
      target: dev
    environment:
      NEXT_PUBLIC_API_URL: http://localhost:8000
      NEXT_PUBLIC_API_KEY: ${API_SHARED_SECRET:-change-me-in-production}
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app:cached
      - /app/node_modules
      - /app/.next
    depends_on:
      api:
        condition: service_healthy
```

**Step 4: Add frontend/.env.local to .gitignore**

Append to `.gitignore`:

```
frontend/.env.local
```

**Step 5: Verify docker compose config is valid**

Run: `docker compose config --quiet`
Expected: No errors

**Step 6: Commit**

```bash
git add frontend/Dockerfile frontend/.env.local compose.yml .gitignore
git commit -m "feat(infra): add frontend Docker service and env configuration"
```

---

### Task 12: Smoke Test

**Step 1: Start the backend**

```bash
docker compose up api -d
```

Wait for healthy status:
```bash
docker compose ps api
```
Expected: Status shows "healthy"

**Step 2: Start the frontend**

```bash
cd frontend && npm run dev
```

**Step 3: Manual verification**

Open `http://localhost:3000` in browser. Verify:
- [ ] Empty state with "Ask your documents anything" headline (Instrument Serif)
- [ ] Suggestion chips visible below headline
- [ ] Sidebar with pipeline selector and rerank toggle
- [ ] Type a query, see streaming response (if backend has documents indexed)
- [ ] Citations accordion works
- [ ] Dark mode toggle works
- [ ] Resize to mobile width, see bottom sheet instead of sidebar

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete frontend chat UI with SSE streaming, citations, responsive design"
```

---

## Summary

| Lane | Tasks | Description |
|------|-------|-------------|
| Lane 1 (Backend) | Tasks 1-3 | SSE schemas, RAGService streaming, streaming router |
| Lane 2 (Frontend) | Tasks 4-10 | Next.js init, theme, API client, components, pages |
| Lane 3 (Integration) | Tasks 11-12 | Docker, env config, smoke test |

**Parallel execution:** Lane 1 (Tasks 1-3) and Lane 2 (Tasks 4-10) can run in parallel worktrees. Lane 3 (Tasks 11-12) runs after both merge.

**Total: 12 tasks, ~45 commits**

**TODO (deferred):**
- Frontend E2E tests with Playwright (tracked in TODOS.md)
