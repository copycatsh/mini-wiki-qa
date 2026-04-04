"""LangGraph RAG implementation"""
import logging
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END

from rag.retrieval import multi_query_retrieve

logger = logging.getLogger(__name__)


class RAGState(TypedDict):
    """State for RAG graph"""
    query: str
    chunks: List[Dict]
    answer: str
    use_rerank: bool
    use_multi_query: bool
    metadata: Dict
    is_safe: bool
    error: str
    history: List[Dict]


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
        """Check query and history for injection attempts"""
        logger.info("Injection Guard node: checking query and history...")

        result = injection_guard.check(state["query"])

        if result["is_safe"]:
            for msg in state.get("history", []):
                hist_result = injection_guard.check(msg["content"])
                if not hist_result["is_safe"]:
                    result = hist_result
                    break

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
        logger.info(f"Retrieve node: query={state['query'][:50]}...")

        top_k = 20 if state.get("use_rerank", False) else 5

        if state.get("use_multi_query", False):
            chunks = multi_query_retrieve(generator.llm, retriever, state["query"], top_k=top_k)
            state["metadata"] = {
                **state.get("metadata", {}),
                "multi_query": True,
                "retrieval_count": len(chunks),
            }
        else:
            chunks = retriever.retrieve(state["query"], top_k=top_k)
            state["metadata"] = {
                **state.get("metadata", {}),
                "retrieval_count": len(chunks),
            }

        state["chunks"] = chunks
        logger.info(f"Retrieved {len(chunks)} chunks")
        return state

    def rerank_node(state: RAGState) -> RAGState:
        """Rerank chunks (conditional)"""
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
        logger.info("Generate node: generating answer...")

        answer = generator.generate(state["query"], state["chunks"], history=state.get("history"))

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
    def route_after_guard(state: RAGState) -> str:
        return "retrieve" if state["is_safe"] else END

    workflow.add_conditional_edges(
        "injection_guard",
        route_after_guard,
        {"retrieve": "retrieve", END: END},
    )
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", "pii_scrubber")
    workflow.add_edge("pii_scrubber", END)

    # Compile
    graph = workflow.compile()

    logger.info("RAG graph created with safety layers")
    return graph
