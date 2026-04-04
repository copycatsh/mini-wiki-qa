"""Pydantic request/response schemas for the Mini-Wiki Q&A API."""

from pydantic import BaseModel, Field
from typing import List, Optional


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="User question")
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    use_rerank: Optional[bool] = Field(default=False, description="Enable reranking")


class Citation(BaseModel):
    document: str = Field(..., description="Source document name")
    chunk_id: str = Field(..., description="Chunk identifier")
    text: str = Field(..., description="Chunk text snippet")
    score: float = Field(..., description="Relevance score")


class AskResponse(BaseModel):
    answer: str = Field(..., description="Generated answer")
    citations: List[Citation] = Field(default=[], description="Retrieved chunks with citations")
    metadata: dict = Field(default={}, description="Request metadata")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall health status")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    services: dict = Field(..., description="Per-service health status")


class IngestResponse(BaseModel):
    status: str = Field(..., description="Ingestion result status")
    documents_loaded: int = Field(..., description="Number of documents processed")
    chunks_created: int = Field(..., description="Number of chunks created")
