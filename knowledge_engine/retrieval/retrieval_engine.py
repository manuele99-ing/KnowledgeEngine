from __future__ import annotations

from knowledge_engine.domain.knowledge.knowledge_base import KnowledgeBase
from knowledge_engine.retrieval.config.retrieval_config import RetrievalConfig
from knowledge_engine.retrieval.debugging.retrieval_debugger import RetrievalDebugger
from knowledge_engine.retrieval.indexing.section_index import SectionIndex
from knowledge_engine.retrieval.nlp.spacy_tokenizer import SpacyTokenizer
from knowledge_engine.retrieval.query.query_processor import QueryProcessor
from knowledge_engine.retrieval.ranking.bm25_retriever import BM25Retriever
from knowledge_engine.retrieval.retrieved_section import RetrievedSection


class RetrievalEngine:
    """
    High-level retrieval pipeline.

        KnowledgeBase
              │
              ▼
        SectionIndex
              │
              ▼
       QueryProcessor
              │
              ▼
      BM25Retriever
              │
              ▼
     SmartSectionRanker
              │
              ▼
      RetrievedSection
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        tokenizer: SpacyTokenizer | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:

        self._config = config or RetrievalConfig()

        tokenizer = tokenizer or SpacyTokenizer()

        self._query_processor = QueryProcessor(
            tokenizer=tokenizer,
        )

        self._section_index = SectionIndex.build(
            knowledge_base=knowledge_base,
            tokenizer=tokenizer,
        )

        self._retriever = BM25Retriever(
            section_index=self._section_index,
            query_processor=self._query_processor,
            k1=self._config.bm25_k1,
            b=self._config.bm25_b,
            semantic_weight=self._config.semantic_weight,
        )

        self._debugger = RetrievalDebugger(
            config=self._config,
        )

    def search(
        self,
        question: str,
        top_k: int | None = None,
    ) -> list[RetrievedSection]:

        if top_k is None:
            top_k = self._config.default_top_k

        matches, query_tokens = self._retriever.search(
            question=question,
            top_k=top_k,
        )

        self._debugger.print(
            question=question,
            query_tokens=query_tokens,
            ranked_sections=[
                (
                    match.section,
                    match.score,
                )
                for match in matches
            ],
        )

        results: list[RetrievedSection] = []

        for match in matches:

            results.append(
                RetrievedSection(
                    document=match.section.document,
                    section=match.section.section,
                    score=match.score,
                )
            )

        return results