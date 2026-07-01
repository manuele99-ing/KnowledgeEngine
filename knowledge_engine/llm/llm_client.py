from __future__ import annotations

from abc import ABC, abstractmethod

from knowledge_engine.llm.llm_request import LLMRequest
from knowledge_engine.llm.llm_response import LLMResponse


class LLMClient(ABC):
    """
    Interface implemented by local LLM runtimes.
    """

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError
