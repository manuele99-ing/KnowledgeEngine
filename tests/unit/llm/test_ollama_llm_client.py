from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from knowledge_engine.llm.llm_client_factory import create_llm_client_from_env
from knowledge_engine.llm.llm_request import LLMRequest
from knowledge_engine.llm.ollama_llm_client import OllamaLLMClient
from knowledge_engine.llm.static_llm_client import StaticLLMClient


class FakeHttpResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self) -> "FakeHttpResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class OllamaLLMClientTest(unittest.TestCase):
    def test_complete_posts_prompt_to_ollama_generate_endpoint(self) -> None:
        captured_requests = []

        def opener(request, timeout):
            captured_requests.append((request, timeout))
            return FakeHttpResponse(
                {
                    "response": "Hub 2 uses Jeweller.",
                    "done": True,
                }
            )

        client = OllamaLLMClient(
            model="llama-test",
            base_url="http://localhost:11434/",
            timeout_seconds=3.0,
            opener=opener,
        )

        response = client.complete(LLMRequest(prompt="Question?"))

        self.assertEqual(response.text, "Hub 2 uses Jeweller.")
        self.assertEqual(response.metadata["backend"], "ollama")
        self.assertEqual(response.metadata["model"], "llama-test")
        self.assertEqual(response.metadata["done"], True)
        self.assertEqual(len(captured_requests), 1)

        request, timeout = captured_requests[0]
        payload = json.loads(request.data.decode("utf-8"))

        self.assertEqual(request.full_url, "http://localhost:11434/api/generate")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(timeout, 3.0)
        self.assertEqual(payload["model"], "llama-test")
        self.assertEqual(payload["prompt"], "Question?")
        self.assertEqual(payload["stream"], False)

    def test_factory_uses_static_client_by_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            client = create_llm_client_from_env()

        self.assertIsInstance(client, StaticLLMClient)

    def test_factory_can_create_ollama_client(self) -> None:
        with patch.dict(
            os.environ,
            {
                "KNOWLEDGE_ENGINE_LLM": "ollama",
                "OLLAMA_MODEL": "llama-test",
            },
            clear=True,
        ):
            client = create_llm_client_from_env()

        self.assertIsInstance(client, OllamaLLMClient)


if __name__ == "__main__":
    unittest.main()
