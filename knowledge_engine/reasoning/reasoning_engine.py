from __future__ import annotations

from knowledge_engine.domain.knowledge.knowledge_base import KnowledgeBase
from knowledge_engine.llm.llm_client import LLMClient
from knowledge_engine.llm.llm_request import LLMRequest
from knowledge_engine.reasoning.answer import Answer
from knowledge_engine.reasoning.knowledge_context_builder import (
    KnowledgeContextBuilder,
)


class ReasoningEngine:
    """
    Answers questions using a KnowledgeBase and an injected LLM client.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        context_builder: KnowledgeContextBuilder | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._context_builder = context_builder or KnowledgeContextBuilder()

    def answer(self, question: str, knowledge_base: KnowledgeBase) -> Answer:
        context = self._context_builder.build(knowledge_base)
        prompt = self._build_prompt(question=question, context=context)
        response = self._llm_client.complete(
            LLMRequest(
                prompt=prompt,
                metadata={
                    "question": question,
                    "knowledge_base_id": knowledge_base.id,
                },
            )
        )

        return Answer(
            question=question,
            text=response.text,
            context=context,
            metadata=response.metadata,
        )

    def _build_prompt(self, question: str, context: str) -> str:
        return (
            "You are a local knowledge engine.\n"
            "Answer using only the knowledge base context below.\n"
            "If the context is insufficient, say that the answer is not known.\n\n"
            "Knowledge base context:\n"
            f"{context}\n\n"
            "Question:\n"
            f"{question}\n\n"
            "Answer:"
        )
