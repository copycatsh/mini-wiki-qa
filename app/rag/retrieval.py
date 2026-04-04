"""Document retrieval from Qdrant vector store"""
import logging
from typing import List, Dict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from core.config import settings

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """Handles semantic search in Qdrant"""

    def __init__(self):
        """Initialize retriever with embeddings and vector store"""
        # Initialize embeddings
        logger.info("Initializing retriever...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Initialize Qdrant client
        self.client = QdrantClient(url=settings.QDRANT_URL)

        # Initialize vector store
        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=settings.QDRANT_COLLECTION,
            embedding=self.embeddings
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve top-k relevant chunks for query

        Args:
            query: User question
            top_k: Number of chunks to retrieve

        Returns:
            List of dicts with chunk text, metadata, and score
        """
        logger.info(f"Retrieving top-{top_k} chunks for query: {query[:50]}...")

        # Semantic search
        results = self.vectorstore.similarity_search_with_score(
            query,
            k=top_k
        )

        # Format results
        chunks = []
        for doc, score in results:
            chunks.append({
                "text": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "score": float(score)
            })

        logger.info(f"Retrieved {len(chunks)} chunks")
        return chunks


# Global retriever instance
_retriever = None


def get_retriever() -> DocumentRetriever:
    """Get or create global retriever instance"""
    global _retriever
    if _retriever is None:
        _retriever = DocumentRetriever()
    return _retriever