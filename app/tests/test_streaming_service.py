"""Tests for RAGService streaming methods"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from services.rag_service import RAGService


@pytest.fixture
def mock_retriever():
    m = MagicMock()
    m.retrieve.return_value = [
        {"text": "Test chunk content", "source": "test/doc.md", "score": 0.95}
    ]
    return m


@pytest.fixture
def mock_generator():
    m = MagicMock()
    m.generate.return_value = "This is a test answer."
    return m


@pytest.fixture
def mock_reranker():
    m = MagicMock()
    m.rerank.return_value = [
        {"text": "Test chunk content", "source": "test/doc.md", "score": 0.95, "rerank_score": 0.99}
    ]
    return m


@pytest.fixture
def rag_service(mock_retriever, mock_generator, mock_reranker):
    return RAGService(mock_retriever, mock_generator, mock_reranker)


class _FakeChunk:
    def __init__(self, content):
        self.content = content


async def _fake_astream(messages):
    for token in ["Hello", " world", "!"]:
        yield _FakeChunk(token)


async def _fake_astream_with_error(messages):
    yield _FakeChunk("partial")
    raise RuntimeError("LLM connection lost")


@pytest.mark.asyncio
async def test_ask_stream_yields_tokens(rag_service, mock_generator):
    mock_generator.llm = MagicMock()
    mock_generator.llm.astream = _fake_astream
    mock_generator.prompt = MagicMock()
    mock_generator.prompt.format_messages.return_value = [{"role": "user", "content": "test"}]

    events = []
    async for event in rag_service.ask_stream("What is Python?"):
        events.append(event)

    token_events = [e for e in events if e[0] == "token"]
    citation_events = [e for e in events if e[0] == "citations"]
    done_events = [e for e in events if e[0] == "done"]

    assert len(token_events) == 3
    assert token_events[0][1] == "Hello"
    assert token_events[1][1] == " world"
    assert token_events[2][1] == "!"
    assert len(citation_events) == 1
    assert len(done_events) == 1


@pytest.mark.asyncio
async def test_ask_stream_with_rerank(rag_service, mock_generator, mock_reranker):
    mock_generator.llm = MagicMock()
    mock_generator.llm.astream = _fake_astream
    mock_generator.prompt = MagicMock()
    mock_generator.prompt.format_messages.return_value = [{"role": "user", "content": "test"}]

    events = []
    async for event in rag_service.ask_stream("What is Python?", use_rerank=True):
        events.append(event)

    mock_reranker.rerank.assert_called_once()


@pytest.mark.asyncio
async def test_ask_stream_error_during_generation(rag_service, mock_generator):
    mock_generator.llm = MagicMock()
    mock_generator.llm.astream = _fake_astream_with_error
    mock_generator.prompt = MagicMock()
    mock_generator.prompt.format_messages.return_value = [{"role": "user", "content": "test"}]

    events = []
    async for event in rag_service.ask_stream("What is Python?"):
        events.append(event)

    error_events = [e for e in events if e[0] == "error"]
    assert len(error_events) == 1
    assert error_events[0][1] == "An internal error occurred"
