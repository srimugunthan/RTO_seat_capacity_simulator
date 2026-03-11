"""Tests for the LLM provider factory — no real API calls made."""
import os
import pytest
from unittest.mock import patch


def test_openai_provider_returns_chat_openai():
    with patch.dict(os.environ, {
        "LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-4o-mini",
        "OPENAI_API_KEY": "sk-fake-key-for-testing",
    }):
        from api.llm import get_llm
        llm = get_llm()
        assert llm.__class__.__name__ == "ChatOpenAI"


def test_anthropic_provider_returns_chat_anthropic():
    with patch.dict(os.environ, {
        "LLM_PROVIDER": "anthropic", "LLM_MODEL": "claude-haiku-4-5-20251001",
        "ANTHROPIC_API_KEY": "sk-ant-fake-key-for-testing",
    }):
        from api.llm import get_llm
        llm = get_llm()
        assert llm.__class__.__name__ == "ChatAnthropic"


def test_groq_provider_returns_chat_groq():
    with patch.dict(os.environ, {
        "LLM_PROVIDER": "groq", "LLM_MODEL": "llama-3.1-8b-instant",
        "GROQ_API_KEY": "gsk_fake-key-for-testing",
    }):
        from api.llm import get_llm
        llm = get_llm()
        assert llm.__class__.__name__ == "ChatGroq"


def test_ollama_provider_returns_chat_ollama():
    # Ollama needs no API key
    with patch.dict(os.environ, {"LLM_PROVIDER": "ollama", "LLM_MODEL": "llama3.1"}):
        from api.llm import get_llm
        llm = get_llm()
        assert llm.__class__.__name__ == "ChatOllama"


def test_unknown_provider_raises_value_error():
    with patch.dict(os.environ, {"LLM_PROVIDER": "unknown_provider"}):
        from api.llm import get_llm
        with pytest.raises(ValueError, match="unknown_provider"):
            get_llm()


def test_default_provider_is_openai():
    """With no LLM_PROVIDER set, defaults to OpenAI."""
    env = {k: v for k, v in os.environ.items() if k not in ("LLM_PROVIDER", "LLM_MODEL")}
    env["OPENAI_API_KEY"] = "sk-fake-key-for-testing"
    with patch.dict(os.environ, env, clear=True):
        from api.llm import get_llm
        llm = get_llm()
        assert llm.__class__.__name__ == "ChatOpenAI"
