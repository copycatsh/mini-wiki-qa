# FastAPI DI, Service Layer, Router Split — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace global singletons with FastAPI lifespan DI, split main.py into routers + service layer, wire the dead /ingest endpoint, fix api_key="dummy" bug.

**Architecture:** Lifespan creates all RAG instances and stores them in app.state. Dependencies read from app.state. graph.py uses factory with closure-based injection. Routers are thin HTTP layers, service layer handles orchestration. Endpoints that call sync rag/ code use `def` (not `async def`) so FastAPI runs them in a threadpool.

**Tech Stack:** FastAPI 0.109, Pydantic 2.11, LangGraph 0.3, pytest + httpx for testing

**Eng Review:** CLEARED — see `/Users/anton/.claude/plans/flickering-napping-platypus.md`

---

## Task 1: Extract Pydantic Schemas

**Files:**
- Create: `app/api/schemas.py`

**Step 1: Create schemas file**

```python
"""Request/Response models for the API"""
from pydantic import BaseModel, Field
from typing import List, Optional


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="User question")
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    use_rerank: Optional[bool] = Field(default=False, description="Enable reranking")


class Citation(BaseModel):
    document: str = Field(..., description="Document name")
    chunk_id: str = Field(..., description="Chunk identifier")
    text: str = Field(..., description="Chunk text")
    score: float = Field(..., description="Relevance score")


class AskResponse(BaseModel):
    answer: str = Field(..., description="Generated answer")
    citations: List[Citation] = Field(default=[], description="Retrieved chunks")
    metadata: dict = Field(default={}, description="Request metadata")


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: dict


class IngestResponse(BaseModel):
    status: str = Field(..., description="Ingestion status")
    documents_loaded: int = Field(..., description="Number of documents loaded")
    chunks_created: int = Field(..., description="Number of chunks created")
```

**Step 2: Verify import works**

Run: `cd app && python -c "from api.schemas import AskRequest, AskResponse, Citation, HealthResponse, IngestResponse; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add app/api/schemas.py
git commit -m "refactor: extract Pydantic schemas to api/schemas.py"
```

---

## Task 2: Create Dependencies Module

**Files:**
- Create: `app/api/dependencies.py`

**Step 1: Create dependencies file**

```python
"""FastAPI dependency injection functions"""
from fastapi import Header, HTTPException, Request

from core.config import settings
from rag.retrieval import DocumentRetriever
from rag.generation import AnswerGenerator
from rag.reranker import DocumentReranker
from rag.safety import PIIScrubber, InjectionGuard


def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    if x_api_key != settings.API_SHARED_SECRET:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return x_api_key


def get_retriever(request: Request) -> DocumentRetriever:
    return request.app.state.retriever


def get_generator(request: Request) -> AnswerGenerator:
    return request.app.state.generator


def get_reranker(request: Request) -> DocumentReranker:
    return request.app.state.reranker


def get_pii_scrubber(request: Request) -> PIIScrubber:
    return request.app.state.pii_scrubber


def get_injection_guard(request: Request) -> InjectionGuard:
    return request.app.state.injection_guard


def get_rag_graph(request: Request):
    return request.app.state.rag_graph
```

**Step 2: Verify import works**

Run: `cd app && python -c "from api.dependencies import verify_api_key, get_retriever, get_generator; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add app/api/dependencies.py
git commit -m "refactor: create FastAPI dependencies module"
```

---

## Task 3: Refactor graph.py — Factory with Closure Injection

**Files:**
- Modify: `app/rag/graph.py`

**Step 1: Rewrite graph.py with factory pattern**

Replace entire file. Key changes:
- `create_rag_graph()` now takes all 5 deps as params
- Node functions are closures inside the factory, capturing injected instances
- Remove `_rag_graph` global and `get_rag_graph()` getter
- Keep `RAGState` TypedDict unchanged

```python
"""LangGraph RAG implementation with dependency injection"""
import logging
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)


class RAGState(TypedDict):
    query: str
    chunks: List[Dict]
    answer: str
    use_rerank: bool
    metadata: Dict
    is_safe: bool
    error: str


def create_rag_graph(retriever, generator, reranker, pii_scrubber, injection_guard):
    """
    Create RAG graph with injected dependencies.

    Args:
        retriever: DocumentRetriever instance
        generator: AnswerGenerator instance
        reranker: DocumentReranker instance
        pii_scrubber: PIIScrubber instance
        injection_guard: InjectionGuard instance

    Returns:
        Compiled LangGraph
    """

    def injection_guard_node(state: RAGState) -> RAGState:
        logger.info("Injection Guard node: checking query...")
        result = injection_guard.check(state["query"])

        state["is_safe"] = result["is_safe"]
        state["metadata"] = {
            **state.get("metadata", {}),
            "injection_check": {
                "is_safe": result["is_safe"],
                "risk_level": result["risk_level"]
            }
        }

        if not result["is_safe"]:
            logger.warning(f"Injection detected: {result['detected_patterns']}")
            state["error"] = "Query blocked: potential injection attempt detected"
            state["answer"] = "Your query was blocked for security reasons. Please rephrase your question."

        return state

    def retrieve_node(state: RAGState) -> RAGState:
        if not state.get("is_safe", True):
            logger.info("Retrieve node: skipped (query blocked)")
            return state

        logger.info(f"Retrieve node: query={state['query'][:50]}...")
        top_k = 20 if state.get("use_rerank", False) else 5
        chunks = retriever.retrieve(state["query"], top_k=top_k)

        state["chunks"] = chunks
        state["metadata"] = {
            **state.get("metadata", {}),
            "retrieval_count": len(chunks)
        }

        logger.info(f"Retrieved {len(chunks)} chunks")
        return state

    def rerank_node(state: RAGState) -> RAGState:
        if not state.get("is_safe", True):
            logger.info("Rerank node: skipped (query blocked)")
            return state

        if not state.get("use_rerank", False):
            logger.info("Rerank node: skipping (use_rerank=False)")
            return state

        logger.info("Rerank node: reranking chunks...")
        chunks = reranker.rerank(state["query"], state["chunks"], top_k=5)

        state["chunks"] = chunks
        state["metadata"] = {
            **state.get("metadata", {}),
            "reranked": True
        }

        logger.info(f"Reranked to {len(chunks)} chunks")
        return state

    def generate_node(state: RAGState) -> RAGState:
        if not state.get("is_safe", True):
            logger.info("Generate node: skipped (query blocked)")
            return state

        logger.info("Generate node: generating answer...")
        answer = generator.generate(state["query"], state["chunks"])

        state["answer"] = answer
        logger.info(f"Generated answer: {answer[:100]}...")
        return state

    def pii_scrubber_node(state: RAGState) -> RAGState:
        if not state.get("answer"):
            logger.info("PII Scrubber node: skipped (no answer)")
            return state

        logger.info("PII Scrubber node: checking for PII...")
        result = pii_scrubber.scrub(state["answer"])

        state["answer"] = result["text"]
        state["metadata"] = {
            **state.get("metadata", {}),
            "pii_scrubbed": {
                "was_scrubbed": result["was_scrubbed"],
                "pii_types": result["pii_detected"]
            }
        }

        if result["was_scrubbed"]:
            logger.warning(f"PII scrubbed from answer: {result['pii_detected']}")

        return state

    workflow = StateGraph(RAGState)

    workflow.add_node("injection_guard", injection_guard_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("pii_scrubber", pii_scrubber_node)

    workflow.set_entry_point("injection_guard")
    workflow.add_edge("injection_guard", "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", "pii_scrubber")
    workflow.add_edge("pii_scrubber", END)

    graph = workflow.compile()

    logger.info("RAG graph created with safety layers")
    return graph
```

**Step 2: Verify import works**

Run: `cd app && python -c "from rag.graph import create_rag_graph, RAGState; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add app/rag/graph.py
git commit -m "refactor: graph.py factory with closure-based dependency injection"
```

---

## Task 4: Remove Global Singletons from rag/ Modules

**Files:**
- Modify: `app/rag/retrieval.py` (delete lines 68-77: `_retriever` global + `get_retriever()`)
- Modify: `app/rag/generation.py` (delete lines 88-96: `_generator` global + `get_generator()`)
- Modify: `app/rag/reranker.py` (delete lines 67-76: `_reranker` global + `get_reranker()`)
- Modify: `app/rag/safety.py` (delete lines 123-140: both globals + both getters)
- Modify: `app/rag/ingest.py` (refactor to accept injected deps)

**Step 1: Remove singleton from retrieval.py**

Delete everything after the `DocumentRetriever` class (lines 68-77):
```python
# DELETE these lines:
# Global retriever instance
_retriever = None

def get_retriever() -> DocumentRetriever:
    ...
```

**Step 2: Remove singleton from generation.py**

Delete everything after the `AnswerGenerator` class (lines 88-96):
```python
# DELETE these lines:
# Global generator instance
_generator = None

def get_generator() -> AnswerGenerator:
    ...
```

**Step 3: Remove singleton from reranker.py**

Delete everything after the `DocumentReranker` class (lines 67-76):
```python
# DELETE these lines:
# Global reranker instance
_reranker = None

def get_reranker() -> DocumentReranker:
    ...
```

**Step 4: Remove singletons from safety.py**

Delete everything after the `InjectionGuard` class (lines 123-140):
```python
# DELETE these lines:
# Global instances
_pii_scrubber = None
_injection_guard = None

def get_pii_scrubber() -> PIIScrubber:
    ...

def get_injection_guard() -> InjectionGuard:
    ...
```

**Step 5: Refactor ingest.py — accept injected deps**

Modify `DocumentIngester.__init__()` to accept `embeddings` and `qdrant_client` as params.
Modify `run_ingestion()` to accept `embeddings`, `qdrant_url`, `collection_name`.

Replace `__init__` method:

```python
def __init__(self, embeddings, qdrant_client):
    """Initialize ingester with pre-built embeddings and Qdrant client"""
    logger.info("Initializing document ingester...")
    self.embeddings = embeddings
    self.qdrant_client = qdrant_client

    self.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
```

Replace `index_documents` method — use the injected embeddings:

```python
def index_documents(self, chunks: List):
    logger.info(f"Indexing {len(chunks)} chunks in Qdrant...")
    self.create_collection()

    QdrantVectorStore.from_documents(
        chunks,
        self.embeddings,
        url=self.qdrant_client._client.rest_uri if hasattr(self.qdrant_client, '_client') else settings.QDRANT_URL,
        collection_name=settings.QDRANT_COLLECTION,
    )

    logger.info("Indexing complete!")
```

Actually, simpler approach — keep `url` and `collection_name` as init params too:

```python
def __init__(self, embeddings, qdrant_client, qdrant_url: str, collection_name: str):
    logger.info("Initializing document ingester...")
    self.embeddings = embeddings
    self.qdrant_client = qdrant_client
    self.qdrant_url = qdrant_url
    self.collection_name = collection_name

    self.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
```

Update `create_collection` to use `self.collection_name`:

```python
def create_collection(self):
    collections = self.qdrant_client.get_collections().collections
    collection_names = [c.name for c in collections]

    if self.collection_name not in collection_names:
        logger.info(f"Creating collection: {self.collection_name}")
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=settings.EMBEDDING_DIM,
                distance=Distance.COSINE
            )
        )
    else:
        logger.info(f"Collection {self.collection_name} already exists")
```

Update `index_documents`:

```python
def index_documents(self, chunks: List):
    logger.info(f"Indexing {len(chunks)} chunks in Qdrant...")
    self.create_collection()

    QdrantVectorStore.from_documents(
        chunks,
        self.embeddings,
        url=self.qdrant_url,
        collection_name=self.collection_name,
    )

    logger.info("Indexing complete!")
```

Update `ingest` to return stats:

```python
def ingest(self, docs_dir: str) -> dict:
    logger.info("Starting ingestion pipeline...")
    documents = self.load_documents(docs_dir)
    chunks = self.chunk_documents(documents)
    self.index_documents(chunks)
    logger.info("Ingestion pipeline complete!")
    return {"documents_loaded": len(documents), "chunks_created": len(chunks)}
```

Update `run_ingestion`:

```python
def run_ingestion(embeddings, qdrant_url: str, collection_name: str, docs_dir: str = None) -> dict:
    if docs_dir is None:
        project_root = Path(__file__).parent.parent.parent
        docs_dir = str(project_root / "data" / "documents" / "squad")

    qdrant_client = QdrantClient(url=qdrant_url)
    ingester = DocumentIngester(embeddings, qdrant_client, qdrant_url, collection_name)
    return ingester.ingest(docs_dir)
```

Remove the `if __name__ == "__main__"` block or update it to construct deps.

**Step 6: Verify all imports still work**

Run: `cd app && python -c "from rag.retrieval import DocumentRetriever; from rag.generation import AnswerGenerator; from rag.reranker import DocumentReranker; from rag.safety import PIIScrubber, InjectionGuard; from rag.ingest import DocumentIngester, run_ingestion; print('OK')"`
Expected: `OK`

**Step 7: Commit**

```bash
git add app/rag/retrieval.py app/rag/generation.py app/rag/reranker.py app/rag/safety.py app/rag/ingest.py
git commit -m "refactor: remove global singletons from all rag/ modules"
```

---

## Task 5: Create Service Layer

**Files:**
- Create: `app/services/__init__.py`
- Create: `app/services/rag_service.py`

**Step 1: Create service module**

`app/services/__init__.py` — empty file.

`app/services/rag_service.py`:

```python
"""RAG service layer — orchestrates retrieval, reranking, and generation"""
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self, retriever, generator, reranker):
        self.retriever = retriever
        self.generator = generator
        self.reranker = reranker

    def ask(self, query: str, top_k: int = 5, use_rerank: bool = False) -> dict:
        logger.info(f"Processing query: {query[:50]}...")

        chunks = self.retriever.retrieve(query, top_k=top_k)

        if use_rerank:
            logger.info("Applying reranking...")
            chunks = self.reranker.rerank(query, chunks, top_k=top_k)

        answer = self.generator.generate(query, chunks)

        citations = [
            {
                "document": chunk["source"].split("/")[-1],
                "chunk_id": f"chunk_{idx}",
                "text": chunk["text"][:200] + "...",
                "score": chunk.get("rerank_score", chunk.get("score", 0.0))
            }
            for idx, chunk in enumerate(chunks)
        ]

        return {
            "answer": answer,
            "citations": citations,
            "chunks_retrieved": len(chunks),
        }

    def ask_graph(self, query: str, top_k: int = 5, use_rerank: bool = False, graph=None) -> dict:
        logger.info(f"[Graph] Processing query: {query[:50]}...")

        result = graph.invoke({
            "query": query,
            "chunks": [],
            "answer": "",
            "use_rerank": use_rerank,
            "metadata": {},
            "is_safe": True,
            "error": "",
        })

        citations = [
            {
                "document": chunk["source"].split("/")[-1],
                "chunk_id": f"chunk_{idx}",
                "text": chunk["text"][:200] + "...",
                "score": chunk.get("rerank_score", chunk.get("score", 0.0))
            }
            for idx, chunk in enumerate(result.get("chunks", []))
        ]

        return {
            "answer": result.get("answer", ""),
            "citations": citations,
            "metadata": result.get("metadata", {}),
        }
```

**Step 2: Verify import works**

Run: `cd app && python -c "from services.rag_service import RAGService; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add app/services/__init__.py app/services/rag_service.py
git commit -m "feat: add RAG service layer"
```

---

## Task 6: Create Routers

**Files:**
- Create: `app/api/routers/__init__.py`
- Create: `app/api/routers/health.py`
- Create: `app/api/routers/ask.py`
- Create: `app/api/routers/admin.py`

**Step 1: Create router package**

`app/api/routers/__init__.py` — empty file.

**Step 2: Create health router**

`app/api/routers/health.py`:

```python
"""Health and root endpoints"""
import logging
from datetime import datetime

import httpx
from fastapi import APIRouter

from api.schemas import HealthResponse
from core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", tags=["Root"])
async def root():
    return {
        "message": "Mini-Wiki Q&A API",
        "docs": "/docs",
        "health": "/health"
    }


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    services = {}

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{settings.QDRANT_URL}/readyz", timeout=2.0)
            services["qdrant"] = "ok" if resp.status_code == 200 else "error"
    except Exception as e:
        services["qdrant"] = f"error: {str(e)}"

    try:
        llm_url = settings.LM_STUDIO_URL if settings.LLM_BACKEND == "lm-studio" else settings.OLLAMA_URL
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{llm_url}/models", timeout=2.0)
            services["llm"] = "ok" if resp.status_code == 200 else "error"
    except Exception as e:
        services["llm"] = f"error: {str(e)}"

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{settings.MLFLOW_TRACKING_URI}/health", timeout=2.0)
            services["mlflow"] = "ok" if resp.status_code == 200 else "error"
    except Exception as e:
        services["mlflow"] = f"error: {str(e)}"

    return HealthResponse(
        status="healthy" if all(v == "ok" for v in services.values()) else "degraded",
        timestamp=datetime.utcnow().isoformat(),
        services=services
    )
```

**Step 3: Create ask router**

`app/api/routers/ask.py`:

```python
"""RAG question-answering endpoints"""
import logging

from fastapi import APIRouter, Depends, HTTPException

from api.schemas import AskRequest, AskResponse, Citation
from api.dependencies import verify_api_key, get_retriever, get_generator, get_reranker, get_rag_graph
from core.config import settings
from services.rag_service import RAGService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ask", response_model=AskResponse, tags=["RAG"])
def ask_question(
    request: AskRequest,
    api_key: str = Depends(verify_api_key),
    retriever=Depends(get_retriever),
    generator=Depends(get_generator),
    reranker=Depends(get_reranker),
):
    try:
        service = RAGService(retriever, generator, reranker)
        result = service.ask(request.query, top_k=request.top_k, use_rerank=request.use_rerank)

        citations = [Citation(**c) for c in result["citations"]]

        return AskResponse(
            answer=result["answer"],
            citations=citations,
            metadata={
                "query": request.query,
                "top_k": request.top_k,
                "use_rerank": request.use_rerank,
                "llm_backend": settings.LLM_BACKEND,
                "chunks_retrieved": result["chunks_retrieved"],
            }
        )

    except Exception as e:
        logger.error(f"Error in /ask: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/ask-graph", response_model=AskResponse, tags=["RAG"])
def ask_question_graph(
    request: AskRequest,
    api_key: str = Depends(verify_api_key),
    graph=Depends(get_rag_graph),
    retriever=Depends(get_retriever),
    generator=Depends(get_generator),
    reranker=Depends(get_reranker),
):
    try:
        service = RAGService(retriever, generator, reranker)
        result = service.ask_graph(
            request.query,
            top_k=request.top_k,
            use_rerank=request.use_rerank,
            graph=graph,
        )

        citations = [Citation(**c) for c in result["citations"]]

        return AskResponse(
            answer=result["answer"],
            citations=citations,
            metadata={
                **result.get("metadata", {}),
                "query": request.query,
                "top_k": request.top_k,
                "use_rerank": request.use_rerank,
                "llm_backend": settings.LLM_BACKEND,
                "pipeline": "langgraph",
            }
        )

    except Exception as e:
        logger.error(f"Error in /ask-graph: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
```

**Step 4: Create admin router**

`app/api/routers/admin.py`:

```python
"""Admin endpoints — ingestion"""
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request

from api.schemas import IngestResponse
from api.dependencies import verify_api_key
from core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse, tags=["Admin"])
def ingest_documents(
    request: Request,
    api_key: str = Depends(verify_api_key),
):
    try:
        from rag.ingest import run_ingestion

        project_root = Path(__file__).parent.parent.parent.parent
        docs_dir = str(project_root / "data" / "documents" / "squad")

        docs_path = Path(docs_dir)
        if not docs_path.exists():
            raise HTTPException(status_code=422, detail=f"Documents directory not found: {docs_dir}")

        md_files = list(docs_path.glob("**/*.md"))
        if not md_files:
            raise HTTPException(status_code=422, detail=f"No markdown documents found in {docs_dir}")

        embeddings = request.app.state.retriever.embeddings
        result = run_ingestion(
            embeddings=embeddings,
            qdrant_url=settings.QDRANT_URL,
            collection_name=settings.QDRANT_COLLECTION,
            docs_dir=docs_dir,
        )

        return IngestResponse(
            status="success",
            documents_loaded=result["documents_loaded"],
            chunks_created=result["chunks_created"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /ingest: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
```

**Step 5: Verify all router imports work**

Run: `cd app && python -c "from api.routers.health import router as h; from api.routers.ask import router as a; from api.routers.admin import router as ad; print('OK')"`
Expected: `OK`

**Step 6: Commit**

```bash
git add app/api/routers/
git commit -m "feat: create health, ask, and admin routers"
```

---

## Task 7: Rewrite main.py — Lifespan + Router Registration

**Files:**
- Modify: `app/api/main.py` (full rewrite)

**Step 1: Rewrite main.py**

```python
"""FastAPI application with lifespan-managed DI"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from rag.retrieval import DocumentRetriever
from rag.generation import AnswerGenerator
from rag.reranker import DocumentReranker
from rag.safety import PIIScrubber, InjectionGuard
from rag.graph import create_rag_graph

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Mini-Wiki Q&A API...")
    logger.info(f"LLM Backend: {settings.LLM_BACKEND}")
    logger.info(f"Qdrant URL: {settings.QDRANT_URL}")
    logger.info(f"MLflow URI: {settings.MLFLOW_TRACKING_URI}")

    app.state.retriever = DocumentRetriever()
    app.state.generator = AnswerGenerator()
    app.state.reranker = DocumentReranker()
    app.state.pii_scrubber = PIIScrubber()
    app.state.injection_guard = InjectionGuard()

    app.state.rag_graph = create_rag_graph(
        retriever=app.state.retriever,
        generator=app.state.generator,
        reranker=app.state.reranker,
        pii_scrubber=app.state.pii_scrubber,
        injection_guard=app.state.injection_guard,
    )

    logger.info("All RAG components initialized")
    yield
    logger.info("Shutting down Mini-Wiki Q&A API...")


app = FastAPI(
    title="Mini-Wiki Q&A API",
    description="RAG-based question answering system",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from api.routers.health import router as health_router
from api.routers.ask import router as ask_router
from api.routers.admin import router as admin_router

app.include_router(health_router)
app.include_router(ask_router)
app.include_router(admin_router)
```

**Step 2: Verify app object creates without error**

Run: `cd app && python -c "from api.main import app; print(f'Routes: {len(app.routes)}')"`
Expected: `Routes: N` (some number > 5)

**Step 3: Commit**

```bash
git add app/api/main.py
git commit -m "refactor: rewrite main.py with lifespan DI and router registration"
```

---

## Task 8: Fix api_key Validation in Config

**Files:**
- Modify: `app/core/config.py`
- Modify: `app/rag/generation.py`

**Step 1: Add model validator to Settings**

In `app/core/config.py`, add import and validator:

```python
"""Application configuration from environment variables"""
from pydantic_settings import BaseSettings
from pydantic import model_validator
from typing import Literal


class Settings(BaseSettings):
    # ... all existing fields unchanged ...

    class Config:
        env_file = ".env"
        case_sensitive = True

    @model_validator(mode='after')
    def validate_openai_key(self):
        if self.LLM_BACKEND == "openai" and not self.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY must be set when LLM_BACKEND is 'openai'. "
                "Set it in your .env file or environment variables."
            )
        return self
```

**Step 2: Remove "dummy" fallback from generation.py**

In `app/rag/generation.py` line 33, change:

```python
# Before:
api_key=settings.OPENAI_API_KEY or "dummy",

# After:
api_key=settings.OPENAI_API_KEY or "not-needed",
```

Note: We use `"not-needed"` instead of empty string because `ChatOpenAI` may reject empty string. Local backends (LM Studio, Ollama) don't validate the key.

**Step 3: Verify validator works**

Run: `cd app && python -c "
import os
os.environ['LLM_BACKEND'] = 'lm-studio'
os.environ['OPENAI_API_KEY'] = ''
from core.config import Settings
s = Settings()
print(f'LM Studio OK: backend={s.LLM_BACKEND}')
"`
Expected: `LM Studio OK: backend=lm-studio`

Run: `cd app && python -c "
import os
os.environ['LLM_BACKEND'] = 'openai'
os.environ['OPENAI_API_KEY'] = ''
try:
    from pydantic_settings import BaseSettings
    from pydantic import model_validator
    from typing import Literal
    class TestSettings(BaseSettings):
        LLM_BACKEND: Literal['lm-studio', 'ollama', 'openai'] = 'openai'
        OPENAI_API_KEY: str = ''
        @model_validator(mode='after')
        def validate_openai_key(self):
            if self.LLM_BACKEND == 'openai' and not self.OPENAI_API_KEY:
                raise ValueError('OPENAI_API_KEY must be set')
            return self
    TestSettings()
    print('ERROR: should have raised')
except Exception as e:
    print(f'Correctly raised: {type(e).__name__}')
"`
Expected: `Correctly raised: ValidationError`

**Step 4: Commit**

```bash
git add app/core/config.py app/rag/generation.py
git commit -m "fix: validate OPENAI_API_KEY when LLM_BACKEND is openai"
```

---

## Task 9: Test Infrastructure — conftest and Health Tests

**Files:**
- Create: `app/tests/__init__.py`
- Create: `app/tests/conftest.py`
- Create: `app/tests/test_health.py`

**Step 1: Create test package and conftest**

`app/tests/__init__.py` — empty file.

`app/tests/conftest.py`:

```python
"""Shared test fixtures"""
import pytest
from unittest.mock import MagicMock, AsyncMock
from fastapi.testclient import TestClient

from api.main import app
from api.dependencies import (
    verify_api_key, get_retriever, get_generator,
    get_reranker, get_pii_scrubber, get_injection_guard, get_rag_graph,
)
from core.config import settings

VALID_API_KEY = settings.API_SHARED_SECRET


@pytest.fixture
def mock_retriever():
    m = MagicMock()
    m.retrieve.return_value = [
        {"text": "Test chunk content", "source": "test/doc.md", "score": 0.95}
    ]
    m.embeddings = MagicMock()
    return m


@pytest.fixture
def mock_generator():
    m = MagicMock()
    m.generate.return_value = "This is a test answer."
    return m


@pytest.fixture
def mock_reranker():
    m = MagicMock()
    m.rerank.return_value = [
        {"text": "Test chunk content", "source": "test/doc.md", "score": 0.95, "rerank_score": 0.99}
    ]
    return m


@pytest.fixture
def mock_pii_scrubber():
    m = MagicMock()
    m.scrub.return_value = {"text": "clean answer", "pii_detected": [], "was_scrubbed": False}
    return m


@pytest.fixture
def mock_injection_guard():
    m = MagicMock()
    m.check.return_value = {"is_safe": True, "detected_patterns": [], "risk_level": "none"}
    return m


@pytest.fixture
def mock_rag_graph():
    m = MagicMock()
    m.invoke.return_value = {
        "query": "test",
        "chunks": [{"text": "Test chunk", "source": "test/doc.md", "score": 0.9}],
        "answer": "Graph answer",
        "use_rerank": False,
        "metadata": {"injection_check": {"is_safe": True, "risk_level": "none"}},
        "is_safe": True,
        "error": "",
    }
    return m


@pytest.fixture
def client(mock_retriever, mock_generator, mock_reranker, mock_pii_scrubber, mock_injection_guard, mock_rag_graph):
    app.dependency_overrides[get_retriever] = lambda: mock_retriever
    app.dependency_overrides[get_generator] = lambda: mock_generator
    app.dependency_overrides[get_reranker] = lambda: mock_reranker
    app.dependency_overrides[get_pii_scrubber] = lambda: mock_pii_scrubber
    app.dependency_overrides[get_injection_guard] = lambda: mock_injection_guard
    app.dependency_overrides[get_rag_graph] = lambda: mock_rag_graph

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()
```

**Step 2: Write health tests**

`app/tests/test_health.py`:

```python
"""Tests for health and root endpoints"""


def test_root_returns_info(client):
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["message"] == "Mini-Wiki Q&A API"
    assert "docs" in data
    assert "health" in data


def test_health_check_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "timestamp" in data
    assert "services" in data
```

**Step 3: Run tests to verify they pass**

Run: `cd app && python -m pytest tests/test_health.py -v`
Expected: 2 tests PASS

**Step 4: Commit**

```bash
git add app/tests/
git commit -m "test: add conftest fixtures and health endpoint tests"
```

---

## Task 10: Tests — Ask Endpoints

**Files:**
- Create: `app/tests/test_ask.py`

**Step 1: Write ask endpoint tests**

```python
"""Tests for /ask and /ask-graph endpoints"""
from tests.conftest import VALID_API_KEY


def test_ask_requires_api_key(client):
    resp = client.post("/ask", json={"query": "What is Python?"})
    assert resp.status_code == 422  # missing header


def test_ask_invalid_api_key(client):
    resp = client.post(
        "/ask",
        json={"query": "What is Python?"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert resp.status_code == 403


def test_ask_happy_path(client, mock_retriever, mock_generator):
    resp = client.post(
        "/ask",
        json={"query": "What is Python?"},
        headers={"X-API-Key": VALID_API_KEY},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert data["answer"] == "This is a test answer."
    assert len(data["citations"]) > 0
    mock_retriever.retrieve.assert_called_once()
    mock_generator.generate.assert_called_once()


def test_ask_with_rerank(client, mock_retriever, mock_reranker, mock_generator):
    resp = client.post(
        "/ask",
        json={"query": "What is Python?", "use_rerank": True},
        headers={"X-API-Key": VALID_API_KEY},
    )
    assert resp.status_code == 200
    mock_reranker.rerank.assert_called_once()


def test_ask_graph_happy_path(client, mock_rag_graph):
    resp = client.post(
        "/ask-graph",
        json={"query": "What is Python?"},
        headers={"X-API-Key": VALID_API_KEY},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "Graph answer"
    assert data["metadata"]["pipeline"] == "langgraph"
    mock_rag_graph.invoke.assert_called_once()
```

**Step 2: Run tests**

Run: `cd app && python -m pytest tests/test_ask.py -v`
Expected: 5 tests PASS

**Step 3: Commit**

```bash
git add app/tests/test_ask.py
git commit -m "test: add ask endpoint tests"
```

---

## Task 11: Tests — Admin and Config

**Files:**
- Create: `app/tests/test_admin.py`
- Create: `app/tests/test_config.py`

**Step 1: Write admin tests**

`app/tests/test_admin.py`:

```python
"""Tests for /ingest endpoint"""
from unittest.mock import patch, MagicMock
from tests.conftest import VALID_API_KEY


def test_ingest_requires_api_key(client):
    resp = client.post("/ingest")
    assert resp.status_code == 422


@patch("api.routers.admin.Path")
@patch("api.routers.admin.run_ingestion")
def test_ingest_happy_path(mock_run, mock_path_cls, client):
    mock_path_instance = MagicMock()
    mock_path_instance.exists.return_value = True
    mock_path_instance.glob.return_value = ["doc1.md", "doc2.md"]
    mock_path_cls.return_value = mock_path_instance
    # Also need to handle the Path(__file__) chain
    mock_path_cls.side_effect = lambda x: mock_path_instance if x != __file__ else MagicMock()

    mock_run.return_value = {"documents_loaded": 2, "chunks_created": 10}

    resp = client.post(
        "/ingest",
        headers={"X-API-Key": VALID_API_KEY},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["documents_loaded"] == 2
    assert data["chunks_created"] == 10
```

**Step 2: Write config tests**

`app/tests/test_config.py`:

```python
"""Tests for config validation"""
import os
import pytest


def test_local_backend_no_key_ok():
    os.environ["LLM_BACKEND"] = "lm-studio"
    os.environ["OPENAI_API_KEY"] = ""
    from core.config import Settings
    s = Settings()
    assert s.LLM_BACKEND == "lm-studio"


def test_openai_backend_requires_api_key():
    os.environ["LLM_BACKEND"] = "openai"
    os.environ["OPENAI_API_KEY"] = ""
    from core.config import Settings
    with pytest.raises(Exception):  # ValidationError
        Settings()
```

**Step 3: Run all tests**

Run: `cd app && python -m pytest tests/ -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add app/tests/test_admin.py app/tests/test_config.py
git commit -m "test: add admin endpoint and config validation tests"
```

---

## Task 12: Integration Test with In-Memory Qdrant

**Files:**
- Create: `app/tests/test_integration.py`

**Step 1: Write integration test**

This test uses real Qdrant (in-memory mode) and real embeddings to verify the full DI wiring end-to-end. It's slower (~10s) but catches real bugs.

```python
"""Integration test — real Qdrant (in-memory) + real embeddings"""
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

from api.main import app
from api.dependencies import (
    verify_api_key, get_retriever, get_generator,
    get_reranker, get_rag_graph,
)
from core.config import settings


@pytest.fixture(scope="module")
def embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@pytest.fixture(scope="module")
def qdrant_with_docs(embeddings):
    client = QdrantClient(":memory:")
    collection_name = "test-integration"

    vector_size = len(embeddings.embed_query("test"))
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    docs = [
        Document(page_content="Python is a programming language created by Guido van Rossum.", metadata={"source": "python.md"}),
        Document(page_content="FastAPI is a modern web framework for building APIs with Python.", metadata={"source": "fastapi.md"}),
    ]

    QdrantVectorStore.from_documents(
        docs, embeddings, location=":memory:", collection_name=collection_name,
    )

    # Return a real vector store for retrieval
    store = QdrantVectorStore(client=client, collection_name=collection_name, embedding=embeddings)
    return store, client


@pytest.fixture
def real_retriever(qdrant_with_docs, embeddings):
    """A real retriever backed by in-memory Qdrant"""
    store, client = qdrant_with_docs
    retriever = MagicMock()

    def real_retrieve(query, top_k=5):
        results = store.similarity_search_with_score(query, k=top_k)
        return [
            {"text": doc.page_content, "source": doc.metadata.get("source", "unknown"), "score": float(score)}
            for doc, score in results
        ]

    retriever.retrieve = real_retrieve
    retriever.embeddings = embeddings
    return retriever


@pytest.fixture
def integration_client(real_retriever):
    mock_generator = MagicMock()
    mock_generator.generate.return_value = "Python is a programming language."

    mock_reranker = MagicMock()
    mock_graph = MagicMock()

    app.dependency_overrides[get_retriever] = lambda: real_retriever
    app.dependency_overrides[get_generator] = lambda: mock_generator
    app.dependency_overrides[get_reranker] = lambda: mock_reranker
    app.dependency_overrides[get_rag_graph] = lambda: mock_graph

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


@pytest.mark.slow
def test_ask_end_to_end(integration_client):
    resp = integration_client.post(
        "/ask",
        json={"query": "What is Python?"},
        headers={"X-API-Key": settings.API_SHARED_SECRET},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert len(data["citations"]) > 0
    assert any("python" in c["text"].lower() for c in data["citations"])
```

**Step 2: Run integration test**

Run: `cd app && python -m pytest tests/test_integration.py -v -m "slow or not slow"`
Expected: 1 test PASS (may take ~10s for model loading)

**Step 3: Commit**

```bash
git add app/tests/test_integration.py
git commit -m "test: add integration test with in-memory Qdrant"
```

---

## Task 13: Docker Build Verification

**Step 1: Build the container**

Run: `docker compose build api`
Expected: Build succeeds with no import errors

**Step 2: Start services and test**

Run: `docker compose up -d api qdrant`
Wait 15 seconds for startup.

Run: `curl -s http://localhost:8000/health | python -m json.tool`
Expected: 200 with qdrant=ok

Run: `curl -s -X POST http://localhost:8000/ask -H "X-API-Key: change-me-in-production" -H "Content-Type: application/json" -d '{"query":"test"}' | python -m json.tool`
Expected: Response (may be error if no LLM running, but should not be 500 from import error)

Run: `curl -s -X POST http://localhost:8000/ask -H "X-API-Key: wrong" -H "Content-Type: application/json" -d '{"query":"test"}'`
Expected: 403 Forbidden

**Step 3: Run test suite in container**

Run: `docker compose exec api python -m pytest tests/ -v --ignore=tests/test_integration.py`
Expected: All unit tests PASS

**Step 4: Commit any final fixes**

```bash
git add -A
git commit -m "chore: final verification fixes"
```

Only if there are actual fixes needed. Skip if all clean.
