"""Answer generation using LLM"""
import logging
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

from core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on provided context.

Rules:
- Answer ONLY based on the provided context
- If the context doesn't contain the answer, say "I don't have enough information to answer this question"
- Be concise and direct
- Cite the document sources when relevant"""


class AnswerGenerator:
    """Generates answers from retrieved context using LLM"""

    def __init__(self):
        """Initialize LLM client"""
        logger.info(f"Initializing generator with backend: {settings.LLM_BACKEND}")

        # Select LLM backend
        if settings.LLM_BACKEND == "lm-studio":
            base_url = settings.LM_STUDIO_URL
            model = settings.LM_STUDIO_MODEL
        elif settings.LLM_BACKEND == "ollama":
            base_url = settings.OLLAMA_URL
            model = settings.OLLAMA_MODEL
        else:  # openai
            base_url = "https://api.openai.com/v1"
            model = "gpt-4"

        self.llm = ChatOpenAI(
            base_url=base_url,
            api_key=settings.OPENAI_API_KEY or "not-needed",
            model=model,
            temperature=0.0,
            max_tokens=500
        )

    def build_messages(
        self, query: str, chunks: List[Dict], history: Optional[List[Dict]] = None
    ) -> List[BaseMessage]:
        """Build LLM message list with optional conversation history.

        Returns [SystemMessage, *history_messages, HumanMessage(context + question)].
        History is truncated to the last MAX_HISTORY_PAIRS pairs.
        """
        messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

        if history:
            max_messages = settings.MAX_HISTORY_PAIRS * 2
            trimmed = history[-max_messages:]
            for msg in trimmed:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))

        context = "\n\n---\n\n".join(
            f"Document: {chunk['source']}\n{chunk['text']}" for chunk in chunks
        )
        messages.append(HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"))
        return messages

    def generate(self, query: str, chunks: List[Dict], history: Optional[List[Dict]] = None) -> str:
        """Generate answer from query, retrieved chunks, and optional history."""
        logger.info(f"Generating answer for query: {query[:50]}...")

        messages = self.build_messages(query, chunks, history)
        response = self.llm.invoke(messages)
        answer = response.content

        logger.info(f"Generated answer: {answer[:100]}...")
        return answer
