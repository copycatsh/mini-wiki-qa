"""RAG service layer — orchestrates retrieval, reranking, and generation"""
import asyncio
import logging
from typing import List, Dict, Optional, AsyncGenerator, Tuple, Any
from core.config import settings

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self, retriever, generator, reranker):
        self.retriever = retriever
        self.generator = generator
        self.reranker = reranker

    def ask(
        self, query: str, top_k: int = 5, use_rerank: bool = False,
        history: Optional[List[Dict]] = None,
    ) -> dict:
        logger.info(f"Processing query: {query[:50]}...")
        chunks = self.retriever.retrieve(query, top_k=top_k)

        if use_rerank:
            logger.info("Applying reranking...")
            chunks = self.reranker.rerank(query, chunks, top_k=top_k)

        answer = self.generator.generate(query, chunks, history=history)

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

    async def ask_stream(
        self, query: str, top_k: int = 5, use_rerank: bool = False,
        history: Optional[List[Dict]] = None,
    ) -> AsyncGenerator[Tuple[str, Any], None]:
        try:
            chunks = await asyncio.to_thread(self.retriever.retrieve, query, top_k=top_k)
            if use_rerank:
                chunks = await asyncio.to_thread(self.reranker.rerank, query, chunks, top_k=top_k)

            messages = self.generator.build_messages(query, chunks, history)

            async for chunk in self.generator.llm.astream(messages):
                if chunk.content:
                    yield ("token", chunk.content)

            citations = [
                {
                    "document": c["source"].split("/")[-1],
                    "chunk_id": f"chunk_{i}",
                    "text": c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"],
                    "score": c.get("rerank_score", c.get("score", 0.0)),
                }
                for i, c in enumerate(chunks)
            ]

            yield ("citations", citations)
            yield ("metadata", {
                "query": query, "top_k": top_k, "use_rerank": use_rerank,
                "chunks_retrieved": len(chunks), "pipeline": "basic",
                "llm_backend": settings.LLM_BACKEND,
            })
            yield ("done", "")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("ask_stream failed for query=%s: %s", query[:80], e, exc_info=True)
            yield ("error", "An internal error occurred")

    async def ask_graph_stream(
        self, query: str, use_rerank: bool = False, graph=None,
        history: Optional[List[Dict]] = None,
    ) -> AsyncGenerator[Tuple[str, Any], None]:
        if graph is None:
            yield ("error", "RAG graph is not initialized")
            return

        try:
            initial_state = {
                "query": query,
                "chunks": [],
                "answer": "",
                "use_rerank": use_rerank,
                "metadata": {},
                "is_safe": True,
                "error": "",
                "history": history or [],
            }

            final_state = await graph.ainvoke(initial_state)

            if not final_state.get("is_safe", True):
                yield ("error", final_state.get("error", "Query blocked by safety filter"))
                return

            answer = final_state.get("answer", "")
            if not answer.strip():
                logger.warning("Graph returned empty answer for query=%s", query[:80])
                yield ("error", "The system could not generate an answer. Please try again.")
                return

            words = answer.split()
            for i, word in enumerate(words):
                token = word if i == 0 else " " + word
                yield ("token", token)

            chunks = final_state.get("chunks", [])
            citations = [
                {
                    "document": c["source"].split("/")[-1],
                    "chunk_id": f"chunk_{i}",
                    "text": c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"],
                    "score": c.get("rerank_score", c.get("score", 0.0)),
                }
                for i, c in enumerate(chunks)
            ]

            yield ("citations", citations)
            metadata = final_state.get("metadata", {})
            metadata.update({"pipeline": "langgraph", "llm_backend": settings.LLM_BACKEND})
            yield ("metadata", metadata)
            yield ("done", "")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("ask_graph_stream failed for query=%s: %s", query[:80], e, exc_info=True)
            yield ("error", "An internal error occurred")

    def ask_graph(
        self, query: str, use_rerank: bool = False, graph=None,
        history: Optional[List[Dict]] = None,
    ) -> dict:
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
            "history": history or [],
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
