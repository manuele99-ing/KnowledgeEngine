from __future__ import annotations

from math import sqrt

from rank_bm25 import BM25Okapi

from knowledge_engine.retrieval.indexing.section_index import SectionIndex
from knowledge_engine.retrieval.nlp.spacy_tokenizer import SpacyTokenizer
from knowledge_engine.retrieval.retrieval_match import RetrievalMatch


class BM25Retriever:
    """
    BM25 lexical retriever with optional semantic reranking.

    The BM25 index is built once during initialization and reused
    for all subsequent searches.
    """

    def __init__(
        self,
        section_index: SectionIndex,
        tokenizer: SpacyTokenizer | None = None,
        k1: float = 1.5,
        b: float = 0.0,
        semantic_weight: float = 0.5,
    ) -> None:

        self._section_index = section_index
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._semantic_weight = semantic_weight

        corpus = [
            list(section.tokens)
            for section in section_index.sections
        ]

        self._bm25 = BM25Okapi(
            corpus,
            k1=k1,
            b=b,
        )

    def _cosine_similarity(
        self,
        vector_a: tuple[float, ...],
        vector_b: tuple[float, ...],
    ) -> float:

        if not vector_a or not vector_b:
            return 0.0

        dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
        norm_a = sqrt(sum(a * a for a in vector_a))
        norm_b = sqrt(sum(b * b for b in vector_b))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def search(
        self,
        question: str,
        top_k: int = 5,
    ) -> tuple[list[RetrievalMatch], list[str]]:

        query_tokens = list(self._tokenizer.tokenize(question))

        scores = self._bm25.get_scores(query_tokens)
        query_vector = self._tokenizer.vector(question)

        ranked: list[tuple[int, float]] = []

        for index, bm25_score in enumerate(scores):
            semantic_score = self._cosine_similarity(
                query_vector,
                self._section_index.sections[index].vector,
            )
            combined_score = bm25_score + self._semantic_weight * semantic_score
            ranked.append((index, combined_score, bm25_score, semantic_score))

        ranked.sort(key=lambda item: item[1], reverse=True)

        if ranked and ranked[0][1] > 0:
            score_threshold = max(0.1, ranked[0][1] * 0.05)
            filtered = [item for item in ranked if item[1] >= score_threshold]
            ranked = filtered or ranked

        matches: list[RetrievalMatch] = []

        for index, score, bm25_score, semantic_score in ranked[:top_k]:
            indexed_section = self._section_index.sections[index]
            matches.append(
                RetrievalMatch(
                    section=indexed_section,
                    score=float(score),
                )
            )

        return matches, query_tokens
