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
