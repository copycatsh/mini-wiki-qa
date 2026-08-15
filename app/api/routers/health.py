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
    return {"message": "Rag Playground API", "docs": "/docs", "health": "/health"}


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
