from __future__ import annotations

from knowledge_engine.domain.knowledge.knowledge_base import KnowledgeBase
from knowledge_engine.llm.llm_client import LLMClient
from knowledge_engine.llm.llm_request import LLMRequest
from knowledge_engine.reasoning.answer import Answer
from knowledge_engine.reasoning.knowledge_context_builder import KnowledgeContextBuilder
from knowledge_engine.reasoning.prompt_builder import PromptBuilder
from knowledge_engine.retrieval.config.retrieval_config import RetrievalConfig
from knowledge_engine.retrieval.nlp.spacy_tokenizer import SpacyTokenizer
from knowledge_engine.retrieval.retrieval_engine import RetrievalEngine


class ReasoningEngine:
    """
    Answers questions using the RetrievalEngine and an LLM.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        retrieval_engine: RetrievalEngine | None = None,
        knowledge_base: KnowledgeBase | None = None,
        tokenizer: SpacyTokenizer | None = None,
        retrieval_config: RetrievalConfig | None = None,
    ) -> None:

        self._llm = llm_client
        self._retrieval = retrieval_engine
        self._knowledge_base = knowledge_base
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._retrieval_config = retrieval_config
        self._prompt_builder = PromptBuilder()
        self._context_builder = KnowledgeContextBuilder()

    def answer(
        self,
        question: str,
        knowledge_base: KnowledgeBase | None = None,
    ) -> Answer:

        retrieval_engine = self._retrieval

        if retrieval_engine is None:
            retrieval_engine = RetrievalEngine(
                knowledge_base=knowledge_base or self._knowledge_base,
                tokenizer=self._tokenizer,
                config=self._retrieval_config,
            )

        if retrieval_engine is None:
            raise ValueError("No retrieval engine available.")

        sections = retrieval_engine.search(question)
        context = self._context_builder.build(sections)

        if not context:
            return Answer(
                question=question,
                text="Non lo so.",
                context="",
            )

        prompt = self._prompt_builder.build(
            question=question,
            context=context,
        )

        response = self._llm.complete(
            LLMRequest(prompt=prompt)
        )

        answer_text = response.text.strip()

        if answer_text == "Non lo so.":
            answer_text = self._extract_answer_from_context(
                question=question,
                context=context,
            ) or answer_text

        return Answer(
            question=question,
            text=answer_text,
            context=context,
            metadata={
                "source_sections": [
                    item.section.title
                    for item in sections
                ],
            },
        )

    def _extract_answer_from_context(
        self,
        question: str,
        context: str,
    ) -> str:
        question_tokens = set(self._tokenizer.tokenize(question))
        best_line = ""
        best_overlap = 0

        for line in context.splitlines():
            stripped = line.strip()

            if not stripped:
                continue

            if stripped.startswith("Document:") or stripped.startswith("Section:"):
                continue

            if stripped == "Content:":
                continue

            line_tokens = set(self._tokenizer.tokenize(stripped))
            overlap = len(question_tokens.intersection(line_tokens))

            if overlap > best_overlap:
                best_overlap = overlap
                best_line = stripped

        if best_overlap >= 2:
            return best_line.rstrip(".") + "."

        return ""
