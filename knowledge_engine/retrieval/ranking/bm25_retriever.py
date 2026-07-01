from __future__ import annotations

from rank_bm25 import BM25Okapi

from knowledge_engine.retrieval.indexing.section_index import SectionIndex
from knowledge_engine.retrieval.retrieved_section import RetrievedSection
from knowledge_engine.retrieval.nlp.spacy_tokenizer import SpacyTokenizer


class BM25Retriever:
    """
    BM25 lexical retriever.

    The BM25 index is built once during initialization and reused
    for all subsequent searches.
    """

    def __init__(
        self,
        section_index: SectionIndex,
        tokenizer: SpacyTokenizer | None = None,
    ) -> None:

        self._section_index = section_index
        self._tokenizer = tokenizer or SpacyTokenizer()

        corpus = [
            list(section.tokens)
            for section in section_index.sections
        ]

        self._bm25 = BM25Okapi(corpus)

    def search(
        self,
        question: str,
        top_k: int = 5,
    ) -> list[RetrievedSection]:

        query_tokens = list(
            self._tokenizer.tokenize(question)
        )

        scores = self._bm25.get_scores(query_tokens)

        ranked = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        )

        results: list[RetrievedSection] = []

        for index, score in ranked[:top_k]:

            section = self._section_index.sections[index]

            results.append(
                RetrievedSection(
                    document=section.document,
                    section=section.section,
                    score=float(score),
                )
            )

        return results