"""SSE streaming endpoints for RAG Q&A"""
import asyncio
import json
import logging
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from api.schemas import AskRequest
from api.dependencies import (
    verify_api_key, get_retriever, get_generator, get_reranker, get_rag_graph,
    get_injection_guard,
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
    except asyncio.CancelledError:
        logger.debug("Client disconnected from SSE stream")
        raise
    except Exception as e:
        logger.error(f"SSE stream error: {e}", exc_info=True)
        payload = json.dumps({"event": "error", "data": "An internal error occurred"})
        yield f"data: {payload}\n\n"


@router.post("/ask/stream", tags=["RAG Streaming"])
async def ask_stream(
    request: AskRequest,
    api_key: str = Depends(verify_api_key),
    retriever=Depends(get_retriever),
    generator=Depends(get_generator),
    reranker=Depends(get_reranker),
    injection_guard=Depends(get_injection_guard),
):
    guard_result = injection_guard.check(request.query)
    if not guard_result["is_safe"]:
        async def blocked_stream():
            yield ("error", "Query blocked by safety filter")
        return StreamingResponse(
            sse_generator(blocked_stream()),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    service = RAGService(retriever, generator, reranker)
    stream = service.ask_stream(request.query, top_k=request.top_k, use_rerank=request.use_rerank)
    return StreamingResponse(
        sse_generator(stream),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
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
    stream = service.ask_graph_stream(request.query, use_rerank=request.use_rerank, graph=graph)
    return StreamingResponse(
        sse_generator(stream),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
