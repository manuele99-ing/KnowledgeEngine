from __future__ import annotations

from knowledge_engine.retrieval.query.analyzed_query import AnalyzedQuery
from knowledge_engine.retrieval.ranking.section_scorer import (
    SectionScorer,
)
from knowledge_engine.retrieval.retrieval_match import RetrievalMatch


class SmartSectionRanker:
    """
    Second-stage structural reranker.

    BM25 maximizes recall.

    This component improves precision by exploiting
    structural information already extracted during
    indexing.
    """

    def __init__(
        self,
    ) -> None:

        self._scorer = SectionScorer()

    def rerank(
        self,
        matches: list[RetrievalMatch],
        query: AnalyzedQuery,
    ) -> list[RetrievalMatch]:

        reranked: list[RetrievalMatch] = []

        for match in matches:

            structural_score = self._scorer.score(
                section=match.section,
                query=query,
            )

            #
            # Structural score becomes progressively
            # more important than the lexical score.
            #

            final_score = (
                match.score * 0.60
                +
                structural_score * 0.40
            )

            reranked.append(
                RetrievalMatch(
                    section=match.section,
                    score=final_score,
                )
            )

        reranked.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        return self.normalize(
            reranked
        )

    @staticmethod
    def normalize(
        matches: list[RetrievalMatch],
    ) -> list[RetrievalMatch]:

        if not matches:
            return matches

        minimum = min(
            match.score
            for match in matches
        )

        maximum = max(
            match.score
            for match in matches
        )

        if maximum == minimum:
            return matches

        normalized: list[RetrievalMatch] = []

        for match in matches:

            normalized.append(
                RetrievalMatch(
                    section=match.section,
                    score=(
                        (match.score - minimum)
                        /
                        (maximum - minimum)
                    ),
                )
            )

        return normalized