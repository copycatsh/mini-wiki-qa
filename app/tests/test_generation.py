"""Tests for AnswerGenerator.build_messages()"""
from unittest.mock import patch, MagicMock
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from rag.generation import AnswerGenerator


def _make_generator():
    with patch("rag.generation.ChatOpenAI"):
        return AnswerGenerator()


def test_build_messages_empty_history():
    gen = _make_generator()
    chunks = [{"source": "doc.md", "text": "some content"}]
    messages = gen.build_messages("What is X?", chunks)

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert "What is X?" in messages[1].content
    assert "some content" in messages[1].content


def test_build_messages_with_history():
    gen = _make_generator()
    chunks = [{"source": "doc.md", "text": "some content"}]
    history = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
    ]
    messages = gen.build_messages("third question", chunks, history=history)

    assert len(messages) == 6  # system + 4 history + user question
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == "first question"
    assert isinstance(messages[2], AIMessage)
    assert messages[2].content == "first answer"
    assert isinstance(messages[3], HumanMessage)
    assert messages[3].content == "second question"
    assert isinstance(messages[4], AIMessage)
    assert messages[4].content == "second answer"
    assert isinstance(messages[5], HumanMessage)
    assert "third question" in messages[5].content


@patch("rag.generation.settings")
def test_build_messages_truncates_history(mock_settings):
    mock_settings.MAX_HISTORY_PAIRS = 2
    mock_settings.LLM_BACKEND = "lm-studio"
    mock_settings.LM_STUDIO_URL = "http://localhost:1234/v1"
    mock_settings.LM_STUDIO_MODEL = "test"
    mock_settings.OPENAI_API_KEY = ""

    with patch("rag.generation.ChatOpenAI"):
        gen = AnswerGenerator()

    chunks = [{"source": "doc.md", "text": "content"}]
    history = [
        {"role": "user", "content": f"q{i}"}
        if i % 2 == 0
        else {"role": "assistant", "content": f"a{i}"}
        for i in range(10)
    ]

    messages = gen.build_messages("final", chunks, history=history)

    # system + 4 history messages (last 2 pairs) + user question = 6
    assert len(messages) == 6
    assert isinstance(messages[0], SystemMessage)
    # Should have the last 4 history messages (indices 6,7,8,9)
    assert messages[1].content == "q6"
    assert messages[2].content == "a7"
    assert messages[3].content == "q8"
    assert messages[4].content == "a9"


def test_build_messages_none_history():
    gen = _make_generator()
    chunks = [{"source": "doc.md", "text": "content"}]
    messages = gen.build_messages("question", chunks, history=None)

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
