"""Document reranking using cross-encoder"""
import logging
from typing import List, Dict
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class DocumentReranker:
    """Rerank retrieved documents using cross-encoder"""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize reranker

        Args:
            model_name: Cross-encoder model name
        """
        logger.info(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)
        logger.info("Reranker loaded successfully")

    def rerank(
            self,
            query: str,
            chunks: List[Dict],
            top_k: int = None
    ) -> List[Dict]:
        """
        Rerank chunks based on query relevance

        Args:
            query: User question
            chunks: Retrieved chunks with text and metadata
            top_k: Number of top chunks to return (None = all)

        Returns:
            Reranked chunks with updated scores
        """
        if not chunks:
            return chunks

        logger.info(f"Reranking {len(chunks)} chunks...")

        # Prepare pairs for cross-encoder
        pairs = [[query, chunk["text"]] for chunk in chunks]

        # Get reranking scores
        scores = self.model.predict(pairs)

        # Update chunks with new scores
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)
            chunk["original_score"] = chunk.get("score", 0.0)

        # Sort by rerank score
        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

        # Limit to top_k if specified
        if top_k:
            reranked = reranked[:top_k]

        logger.info(f"Reranking complete, returning top-{len(reranked)} chunks")
        return reranked
