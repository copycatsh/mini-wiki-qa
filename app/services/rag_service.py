"""RAG service layer — orchestrates retrieval, reranking, and generation"""
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self, retriever, generator, reranker):
        self.retriever = retriever
        self.generator = generator
        self.reranker = reranker

    def ask(self, query: str, top_k: int = 5, use_rerank: bool = False) -> dict:
        logger.info(f"Processing query: {query[:50]}...")
        chunks = self.retriever.retrieve(query, top_k=top_k)

        if use_rerank:
            logger.info("Applying reranking...")
            chunks = self.reranker.rerank(query, chunks, top_k=top_k)

        answer = self.generator.generate(query, chunks)

        citations = [
            {
                "document": chunk["source"].split("/")[-1],
                "chunk_id": f"chunk_{idx}",
                "text": chunk["text"][:200] + "...",
                "score": chunk.get("rerank_score", chunk.get("score", 0.0))
            }
            for idx, chunk in enumerate(chunks)
        ]

        return {
            "answer": answer,
            "citations": citations,
            "chunks_retrieved": len(chunks),
        }

    def ask_graph(self, query: str, use_rerank: bool = False, graph=None) -> dict:
        if graph is None:
            raise ValueError("RAG graph is not initialized")
        logger.info(f"[Graph] Processing query: {query[:50]}...")

        result = graph.invoke({
            "query": query,
            "chunks": [],
            "answer": "",
            "use_rerank": use_rerank,
            "metadata": {},
            "is_safe": True,
            "error": "",
        })

        citations = [
            {
                "document": chunk["source"].split("/")[-1],
                "chunk_id": f"chunk_{idx}",
                "text": chunk["text"][:200] + "...",
                "score": chunk.get("rerank_score", chunk.get("score", 0.0))
            }
            for idx, chunk in enumerate(result.get("chunks", []))
        ]

        return {
            "answer": result.get("answer", ""),
            "citations": citations,
            "metadata": result.get("metadata", {}),
        }
