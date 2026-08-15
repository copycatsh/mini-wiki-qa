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
    logger.info("Starting Rag Playground API...")
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
    logger.info("Shutting down Rag Playground API...")


app = FastAPI(
    title="Rag Playground API",
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
from api.routers.stream import router as stream_router

app.include_router(health_router)
app.include_router(ask_router)
app.include_router(admin_router)
app.include_router(stream_router)
