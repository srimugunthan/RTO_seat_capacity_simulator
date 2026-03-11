from langchain_core.language_models import BaseChatModel
from .config import get_llm_provider, get_llm_model


def get_llm() -> BaseChatModel:
    """Return a LangChain BaseChatModel based on LLM_PROVIDER and LLM_MODEL env vars."""
    provider = get_llm_provider()
    model = get_llm_model()

    match provider:
        case "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model, temperature=0)

        case "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=model, temperature=0)

        case "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(model=model, temperature=0)

        case "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(model=model)

        case _:
            raise ValueError(
                f"Unknown LLM provider: '{provider}'. "
                "Choose from: openai, anthropic, groq, ollama"
            )
