"""Pydantic request/response schemas for the Mini-Wiki Q&A API."""

from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"] = Field(..., description="Message sender role")
    content: str = Field(..., min_length=1, max_length=2000, description="Message content")


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="User question")
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    use_rerank: Optional[bool] = Field(default=False, description="Enable reranking")
    history: List[HistoryMessage] = Field(default=[], max_length=20, description="Conversation history")


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
