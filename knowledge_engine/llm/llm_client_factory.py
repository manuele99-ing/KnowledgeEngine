from __future__ import annotations

import os

from knowledge_engine.llm.llm_client import LLMClient
from knowledge_engine.llm.ollama_llm_client import OllamaLLMClient
from knowledge_engine.llm.static_llm_client import StaticLLMClient


def create_llm_client_from_env() -> LLMClient:
    """
    Creates an LLM client from environment configuration.
    """

    provider = os.getenv("KNOWLEDGE_ENGINE_LLM", "static").casefold()

    if provider == "ollama":
        return OllamaLLMClient(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    return StaticLLMClient(
        response_text="This is a placeholder answer from the test LLM."
    )
