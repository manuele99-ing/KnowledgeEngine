from __future__ import annotations

from knowledge_engine.llm.llm_client import LLMClient
from knowledge_engine.llm.llm_request import LLMRequest
from knowledge_engine.reasoning.prompt_builder import PromptBuilder
from knowledge_engine.retrieval.retrieval_engine import RetrievalEngine


class ReasoningEngine:
    """
    Answers questions using the RetrievalEngine and an LLM.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        retrieval_engine: RetrievalEngine,
    ) -> None:

        self._llm = llm_client
        self._retrieval = retrieval_engine
        self._prompt_builder = PromptBuilder()

    def answer(
        self,
        question: str,
    ) -> str:

        sections = self._retrieval.search(question)

        prompt = self._prompt_builder.build(
            question=question,
            sections=sections,
        )

        response = self._llm.complete(
            LLMRequest(prompt=prompt)
        )

        return response.text