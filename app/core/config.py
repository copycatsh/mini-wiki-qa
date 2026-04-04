"""Application configuration from environment variables"""
from pydantic_settings import BaseSettings
from pydantic import model_validator
from typing import Literal


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # API
    API_SHARED_SECRET: str = "change-me-in-production"
    LOG_LEVEL: str = "INFO"

    # LLM Backend
    LLM_BACKEND: Literal["lm-studio", "ollama", "openai"] = "lm-studio"
    LM_STUDIO_URL: str = "http://host.docker.internal:1234/v1"
    LM_STUDIO_MODEL: str = "phi-3-mini-4k-instruct"
    OLLAMA_URL: str = "http://ollama:11434/v1"
    OLLAMA_MODEL: str = "phi3"
    OPENAI_API_KEY: str = ""

    # Vector DB
    QDRANT_URL: str = "http://qdrant:6333"
    QDRANT_COLLECTION: str = "mini-wiki"

    # Embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    # Storage
    MINIO_ENDPOINT: str = "minio:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "rag-data"

    # MLflow
    MLFLOW_TRACKING_URI: str = "http://mlflow:5000"
    MLFLOW_EXPERIMENT_NAME: str = "mini-wiki-rag"

    # RAG
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 5

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


# Global settings instance
settings = Settings()