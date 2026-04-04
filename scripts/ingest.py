"""CLI wrapper for document ingestion pipeline.

Usage: python scripts/ingest.py [docs_dir]
Requires Qdrant to be running.
"""
import sys
from pathlib import Path

# Add app/ to Python path so we can import project modules
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from langchain_huggingface import HuggingFaceEmbeddings
from core.config import settings
from rag.ingest import run_ingestion


def main():
    docs_dir = sys.argv[1] if len(sys.argv) > 1 else None

    print(f"Building embeddings model: {settings.EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"Qdrant: {settings.QDRANT_URL}, collection: {settings.QDRANT_COLLECTION}")
    result = run_ingestion(embeddings, settings.QDRANT_URL, settings.QDRANT_COLLECTION, docs_dir)

    print(f"Done: {result['documents_loaded']} documents, {result['chunks_created']} chunks")


if __name__ == "__main__":
    main()
