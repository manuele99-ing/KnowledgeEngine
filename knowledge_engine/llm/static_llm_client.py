from __future__ import annotations

from knowledge_engine.llm.llm_client import LLMClient
from knowledge_engine.llm.llm_request import LLMRequest
from knowledge_engine.llm.llm_response import LLMResponse


class StaticLLMClient(LLMClient):
    """
    Deterministic LLM test double.
    """

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)

        return LLMResponse(text=self._response_text)
