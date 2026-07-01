from __future__ import annotations

import json
from typing import Any, Protocol
from urllib import request as urllib_request

from knowledge_engine.llm.llm_client import LLMClient
from knowledge_engine.llm.llm_request import LLMRequest
from knowledge_engine.llm.llm_response import LLMResponse


class UrlOpener(Protocol):
    def __call__(
        self,
        request: urllib_request.Request,
        timeout: float,
    ) -> Any:
        ...


class OllamaLLMClient(LLMClient):
    """
    LLM client for a local Ollama server.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 120.0,
        opener: UrlOpener | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._opener = opener or urllib_request.urlopen

    def complete(self, request: LLMRequest) -> LLMResponse:
        payload = {
            "model": self._model,
            "prompt": request.prompt,
            "stream": False,
        }
        http_request = urllib_request.Request(
            url=f"{self._base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with self._opener(
            http_request,
            timeout=self._timeout_seconds,
        ) as response:
            response_payload = json.loads(response.read().decode("utf-8"))

        return LLMResponse(
            text=response_payload.get("response", ""),
            metadata={
                "backend": "ollama",
                "model": self._model,
                "done": response_payload.get("done"),
            },
        )
