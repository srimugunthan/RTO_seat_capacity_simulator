import os
from dotenv import load_dotenv

load_dotenv()


def get_llm_provider() -> str:
    return os.getenv("LLM_PROVIDER", "openai").lower()


def get_llm_model() -> str:
    return os.getenv("LLM_MODEL", "gpt-4o-mini")


def get_default_backend() -> str:
    return os.getenv("DEFAULT_BACKEND", "monte_carlo")
