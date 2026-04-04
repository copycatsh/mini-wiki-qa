"""Tests for config validation"""
import pytest
from pydantic import ValidationError


def test_local_backend_no_key_ok(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "lm-studio")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    from core.config import Settings
    s = Settings()
    assert s.LLM_BACKEND == "lm-studio"


def test_ask_request_multi_query_defaults_false():
    from api.schemas import AskRequest
    req = AskRequest(query="test")
    assert req.use_multi_query is False


def test_openai_backend_requires_api_key(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    from core.config import Settings
    with pytest.raises(ValidationError):
        Settings()
