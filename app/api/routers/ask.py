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
    except HTTPException:
        raise
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
        result = service.ask_graph(request.query, use_rerank=request.use_rerank, graph=graph)
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /ask-graph: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
