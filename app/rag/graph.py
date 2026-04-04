"""LangGraph RAG implementation"""
import logging
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)


class RAGState(TypedDict):
    """State for RAG graph"""
    query: str
    chunks: List[Dict]
    answer: str
    use_rerank: bool
    metadata: Dict
    is_safe: bool
    error: str


def create_rag_graph(retriever, generator, reranker, pii_scrubber, injection_guard) -> StateGraph:
    """
    Create RAG graph with safety layers

    Args:
        retriever: DocumentRetriever instance
        generator: AnswerGenerator instance
        reranker: DocumentReranker instance
        pii_scrubber: PIIScrubber instance
        injection_guard: InjectionGuard instance

    Returns:
        Compiled LangGraph
    """

    def injection_guard_node(state: RAGState) -> RAGState:
        """Check query for injection attempts"""
        logger.info("Injection Guard node: checking query...")

        result = injection_guard.check(state["query"])

        state["is_safe"] = result["is_safe"]
        state["metadata"] = {
            **state.get("metadata", {}),
            "injection_check": {
                "is_safe": result["is_safe"],
                "risk_level": result["risk_level"]
            }
        }

        if not result["is_safe"]:
            logger.warning(f"Injection detected: {result['detected_patterns']}")
            state["error"] = "Query blocked: potential injection attempt detected"
            state["answer"] = "Your query was blocked for security reasons. Please rephrase your question."

        return state

    def retrieve_node(state: RAGState) -> RAGState:
        """Retrieve relevant chunks"""
        if not state.get("is_safe", True):
            logger.info("Retrieve node: skipped (query blocked)")
            return state

        logger.info(f"Retrieve node: query={state['query'][:50]}...")

        top_k = 20 if state.get("use_rerank", False) else 5
        chunks = retriever.retrieve(state["query"], top_k=top_k)

        state["chunks"] = chunks
        state["metadata"] = {
            **state.get("metadata", {}),
            "retrieval_count": len(chunks)
        }

        logger.info(f"Retrieved {len(chunks)} chunks")
        return state

    def rerank_node(state: RAGState) -> RAGState:
        """Rerank chunks (conditional)"""
        if not state.get("is_safe", True):
            logger.info("Rerank node: skipped (query blocked)")
            return state

        if not state.get("use_rerank", False):
            logger.info("Rerank node: skipping (use_rerank=False)")
            return state

        logger.info("Rerank node: reranking chunks...")

        chunks = reranker.rerank(state["query"], state["chunks"], top_k=5)

        state["chunks"] = chunks
        state["metadata"] = {
            **state.get("metadata", {}),
            "reranked": True
        }

        logger.info(f"Reranked to {len(chunks)} chunks")
        return state

    def generate_node(state: RAGState) -> RAGState:
        """Generate answer from chunks"""
        if not state.get("is_safe", True):
            logger.info("Generate node: skipped (query blocked)")
            return state

        logger.info("Generate node: generating answer...")

        answer = generator.generate(state["query"], state["chunks"])

        state["answer"] = answer
        logger.info(f"Generated answer: {answer[:100]}...")

        return state

    def pii_scrubber_node(state: RAGState) -> RAGState:
        """Scrub PII from answer"""
        if not state.get("answer"):
            logger.info("PII Scrubber node: skipped (no answer)")
            return state

        logger.info("PII Scrubber node: checking for PII...")

        result = pii_scrubber.scrub(state["answer"])

        state["answer"] = result["text"]
        state["metadata"] = {
            **state.get("metadata", {}),
            "pii_scrubbed": {
                "was_scrubbed": result["was_scrubbed"],
                "pii_types": result["pii_detected"]
            }
        }

        if result["was_scrubbed"]:
            logger.warning(f"PII scrubbed from answer: {result['pii_detected']}")

        return state

    # Create graph
    workflow = StateGraph(RAGState)

    # Add nodes
    workflow.add_node("injection_guard", injection_guard_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("pii_scrubber", pii_scrubber_node)

    # Add edges (safety first!)
    workflow.set_entry_point("injection_guard")
    workflow.add_edge("injection_guard", "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", "pii_scrubber")
    workflow.add_edge("pii_scrubber", END)

    # Compile
    graph = workflow.compile()

    logger.info("RAG graph created with safety layers")
    return graph
