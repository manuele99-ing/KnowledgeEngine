from knowledge_engine.config.settings import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
)
from knowledge_engine.llm.llm_client import LLMClient
from knowledge_engine.llm.ollama_llm_client import OllamaLLMClient


def create_llm_client() -> LLMClient:
    """
    Creates the default LLM client.
    """

    return OllamaLLMClient(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        timeout_seconds=OLLAMA_TIMEOUT_SECONDS,
    )