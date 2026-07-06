from __future__ import annotations

from math import sqrt

from rank_bm25 import BM25Okapi

from knowledge_engine.retrieval.indexing.section_index import SectionIndex
from knowledge_engine.retrieval.nlp.spacy_tokenizer import SpacyTokenizer
from knowledge_engine.retrieval.ranking.smart_section_ranker import (
    SmartSectionRanker,
)
from knowledge_engine.retrieval.retrieval_match import RetrievalMatch


class BM25Retriever:
    """
    BM25 lexical retriever.

    Responsibilities:

    - maximize recall
    - compute lexical score
    - compute semantic similarity
    - delegate precision improvements to SmartSectionRanker
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

        self._ranker = SmartSectionRanker()

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

        dot = sum(a * b for a, b in zip(vector_a, vector_b))

        norm_a = sqrt(sum(a * a for a in vector_a))
        norm_b = sqrt(sum(b * b for b in vector_b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def search(
        self,
        question: str,
        top_k: int = 5,
    ) -> tuple[list[RetrievalMatch], list[str]]:

        query_tokens = list(
            self._tokenizer.tokenize(question)
        )

        bm25_scores = self._bm25.get_scores(query_tokens)

        query_vector = self._tokenizer.vector(question)

        matches: list[RetrievalMatch] = []

        for index, bm25_score in enumerate(bm25_scores):

            indexed_section = self._section_index.sections[index]

            semantic_score = self._cosine_similarity(
                query_vector,
                indexed_section.vector,
            )

            score = (
                bm25_score
                + semantic_score * self._semantic_weight
            )

            if score <= 0:
                continue

            matches.append(
                RetrievalMatch(
                    section=indexed_section,
                    score=float(score),
                )
            )

        matches.sort(
            key=lambda match: match.score,
            reverse=True,
        )

        #
        # Recall phase completed.
        # Precision phase starts here.
        #

        matches = self._ranker.rerank(
            matches[:20],
            query_tokens,
        )

        #
        # Reject completely unrelated questions.
        #

        if matches:

            best = matches[0].score

            if best < 0.75:
                return [], query_tokens

            threshold = max(
                0.25,
                best * 0.10,
            )

            matches = [
                match
                for match in matches
                if match.score >= threshold
            ]

        return matches[:top_k], query_tokens