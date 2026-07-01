from __future__ import annotations

from knowledge_engine.domain.knowledge.knowledge_base import KnowledgeBase
from knowledge_engine.retrieval.indexing.section_index import SectionIndex
from knowledge_engine.retrieval.retrieved_section import RetrievedSection
from knowledge_engine.retrieval.nlp.spacy_tokenizer import SpacyTokenizer
from knowledge_engine.retrieval.ranking.bm25_retriever import BM25Retriever


class RetrievalEngine:
    """
    High-level interface for document retrieval.

    This class encapsulates the indexing process and the retrieval
    algorithms used to search the KnowledgeBase.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        tokenizer: SpacyTokenizer | None = None,
    ) -> None:

        tokenizer = tokenizer or SpacyTokenizer()

        self._section_index = SectionIndex.build(
            knowledge_base=knowledge_base,
            tokenizer=tokenizer,
        )

        self._retriever = BM25Retriever(
            section_index=self._section_index,
            tokenizer=tokenizer,
        )

    def search(
        self,
        question: str,
        top_k: int = 5,
    ) -> list[RetrievedSection]:

        return self._retriever.search(
            question=question,
            top_k=top_k,
        )