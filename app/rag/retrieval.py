"""Document retrieval from Qdrant vector store"""
import hashlib
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from core.config import settings

logger = logging.getLogger(__name__)

_EXPAND_PROMPT = (
    "Generate {num_queries} alternative phrasings of the following question.\n"
    "Each rephrasing should approach the question from a different angle to find\n"
    "relevant information that the original phrasing might miss.\n"
    "Return ONLY the alternative questions, one per line. No numbering, no bullets.\n"
    "\nQuestion: {query}"
)

_NUMBERED_RE = re.compile(r"^\d+[\.\)]\s*")
_BULLET_RE = re.compile(r"^[-*•]\s*")


def expand_queries(llm, query: str, num_queries: int = 3) -> List[str]:
    """Use LLM to generate alternative phrasings. Returns [original] + variants."""
    prompt = _EXPAND_PROMPT.format(num_queries=num_queries, query=query)
    response = llm.invoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)

    variants = []
    for line in text.strip().split("\n"):
        line = line.strip()
        line = _NUMBERED_RE.sub("", line)
        line = _BULLET_RE.sub("", line)
        line = line.strip()
        if line:
            variants.append(line)

    if not variants:
        logger.warning("LLM returned no query variants for: %s", query[:50])
        return [query]

    return [query] + variants


def deduplicate_chunks(chunks: List[Dict], top_k: int = 5) -> List[Dict]:
    """Dedup by text content hash, keep highest score, trim to top_k."""
    seen: Dict[str, Dict] = {}
    for chunk in chunks:
        key = hashlib.md5(chunk["text"].encode()).hexdigest()
        existing = seen.get(key)
        if existing is None or chunk.get("score", 0.0) > existing.get("score", 0.0):
            seen[key] = chunk
    result = sorted(seen.values(), key=lambda c: c.get("score", 0.0), reverse=True)
    return result[:top_k]


def multi_query_retrieve(
    llm, retriever, query: str, top_k: int = 5, num_queries: int = 3,
) -> List[Dict]:
    """Expand queries, retrieve in parallel, dedup. Falls back on LLM error."""
    try:
        queries = expand_queries(llm, query, num_queries)
        logger.info("Query expansion generated %d variants for: %s", len(queries) - 1, query[:50])
    except Exception:
        logger.warning("Query expansion failed, falling back to single query", exc_info=True)
        return retriever.retrieve(query, top_k=top_k)

    per_variant_k = top_k * 2
    all_chunks: List[Dict] = []

    with ThreadPoolExecutor(max_workers=len(queries)) as pool:
        futures = {
            pool.submit(retriever.retrieve, q, per_variant_k): q for q in queries
        }
        try:
            for future in as_completed(futures, timeout=30):
                try:
                    all_chunks.extend(future.result())
                except Exception:
                    failed_q = futures[future]
                    logger.warning("Retrieval failed for variant: %s", failed_q[:50], exc_info=True)
        except TimeoutError:
            logger.warning(
                "Multi-query retrieval timed out; using %d chunks from %d/%d completed queries",
                len(all_chunks), sum(f.done() for f in futures), len(futures),
            )

    if not all_chunks:
        logger.warning("All multi-query variants failed, falling back to single query")
        return retriever.retrieve(query, top_k=top_k)

    return deduplicate_chunks(all_chunks, top_k=top_k)


class DocumentRetriever:
    """Handles semantic search in Qdrant"""

    def __init__(self):
        """Initialize retriever with embeddings and Qdrant client"""
        logger.info("Initializing retriever...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        self.client = QdrantClient(url=settings.QDRANT_URL)
        self._vectorstore = None

    @property
    def vectorstore(self):
        if self._vectorstore is None:
            self._vectorstore = QdrantVectorStore(
                client=self.client,
                collection_name=settings.QDRANT_COLLECTION,
                embedding=self.embeddings
            )
        return self._vectorstore

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
