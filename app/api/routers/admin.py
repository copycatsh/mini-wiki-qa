"""Admin endpoints — ingestion"""
import logging
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from api.schemas import IngestResponse
from api.dependencies import verify_api_key, get_retriever
from core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ingest", response_model=IngestResponse, tags=["Admin"])
def ingest_documents(
    api_key: str = Depends(verify_api_key),
    retriever=Depends(get_retriever),
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

        embeddings = retriever.embeddings
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
