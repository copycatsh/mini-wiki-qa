"""Answer generation using LLM"""
import logging
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from core.config import settings

logger = logging.getLogger(__name__)


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

        # Initialize LLM (OpenAI-compatible)
        self.llm = ChatOpenAI(
            base_url=base_url,
            api_key=settings.OPENAI_API_KEY or "not-needed",
            model=model,
            temperature=0.0,
            max_tokens=500
        )

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based on provided context.

Rules:
- Answer ONLY based on the provided context
- If the context doesn't contain the answer, say "I don't have enough information to answer this question"
- Be concise and direct
- Cite the document sources when relevant"""),
            ("user", """Context:
{context}

Question: {question}

Answer:""")
        ])

    def generate(self, query: str, chunks: List[Dict]) -> str:
        """
        Generate answer from query and retrieved chunks

        Args:
            query: User question
            chunks: Retrieved chunks with text and metadata

        Returns:
            Generated answer
        """
        logger.info(f"Generating answer for query: {query[:50]}...")

        # Format context from chunks
        context = "\n\n---\n\n".join([
            f"Document: {chunk['source']}\n{chunk['text']}"
            for chunk in chunks
        ])

        # Generate answer
        messages = self.prompt.format_messages(
            context=context,
            question=query
        )

        response = self.llm.invoke(messages)
        answer = response.content

        logger.info(f"Generated answer: {answer[:100]}...")
        return answer
