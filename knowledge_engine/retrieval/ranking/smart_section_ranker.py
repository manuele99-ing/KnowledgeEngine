from __future__ import annotations

from dataclasses import dataclass

from knowledge_engine.retrieval.query.analyzed_query import (
    AnalyzedQuery,
)
from knowledge_engine.retrieval.ranking.section_scorer import (
    SectionScorer,
)
from knowledge_engine.retrieval.retrieval_match import (
    RetrievalMatch,
)


@dataclass(slots=True)
class RankingBreakdown:
    """
    Internal representation of the ranking process.

    It keeps every signal separated so that the
    final ranking can evolve without modifying the
    scorer implementation.
    """

    lexical: float

    structural: float

    semantic: float

    entity: float

    intent: float

    keyword: float

    coverage: float

    complete: float

    descriptive: float = 0.0

    procedure: float = 0.0

    final: float = 0.0


class SmartSectionRanker:
    """
    Hybrid second-stage reranker.

    BM25 is responsible for recall.

    This component combines lexical, semantic and
    structural signals into a single final ranking.

    The implementation is intentionally domain
    independent.

    No business knowledge is contained here.
    """

    #
    # -------------------------------------------------------
    # Global weights
    # -------------------------------------------------------
    #

    LEXICAL_WEIGHT = 0.30

    SEMANTIC_WEIGHT = 0.30

    STRUCTURAL_WEIGHT = 0.15

    ENTITY_WEIGHT = 0.10

    INTENT_WEIGHT = 0.07

    KEYWORD_WEIGHT = 0.04

    COVERAGE_WEIGHT = 0.02

    COMPLETE_MATCH_WEIGHT = 0.02

    DESCRIPTIVE_TITLE_WEIGHT = 0.06

    PROCEDURE_TITLE_WEIGHT = 0.08

    def __init__(
        self,
    ) -> None:

        self._scorer = SectionScorer()

    #
    # -------------------------------------------------------
    # Public API
    # -------------------------------------------------------
    #

    def rerank(
        self,
        matches: list[RetrievalMatch],
        query: AnalyzedQuery,
    ) -> list[RetrievalMatch]:

        reranked: list[
            tuple[
                RetrievalMatch,
                RankingBreakdown,
            ]
        ] = []

        for match in matches:

            breakdown = self._compute_breakdown(
                match,
                query,
            )

            reranked.append(
                (
                    RetrievalMatch(
                        section=match.section,
                        score=breakdown.final,
                    ),
                    breakdown,
                )
            )

        reranked.sort(
            key=lambda item: item[0].score,
            reverse=True,
        )

        return self.normalize(

            [
                match
                for match, _
                in reranked
            ]

        )


    #
    # -------------------------------------------------------
    # Hybrid Score Fusion
    # -------------------------------------------------------
    #

    def _compute_breakdown(
        self,
        match: RetrievalMatch,
        query: AnalyzedQuery,
    ) -> RankingBreakdown:

        #
        # Structural analysis.
        #

        structural = self._scorer.debug_score(
            section=match.section,
            query=query,
        )

        #
        # First-stage score.
        #
        # Produced by BM25Retriever:
        #
        #     BM25
        #   + cosine similarity
        #

        lexical_score = match.score

        #
        # Language confidence.
        #

        semantic_confidence = max(
            query.semantic_confidence,
            query.confidence,
        )

        #
        # Intent confidence should never dominate
        # the ranking if Language is uncertain.
        #

        intent_score = (
            structural["intent"]
            *
            semantic_confidence
        )

        #
        # Entity confidence.
        #

        entity_score = structural["entity"]

        #
        # Semantic keyword contribution.
        #

        keyword_score = structural[
            "semantic_keyword"
        ]

        coverage_score = structural[
            "coverage"
        ]

        complete_score = structural[
            "complete_match"
        ]

        descriptive_score = structural.get(
            "descriptive_title",
            0.0,
        )

        procedure_score = structural.get(
            "procedure_title",
            0.0,
        )

        #
        # Aggregate structural score.
        #

        structural_score = (

            intent_score

            + entity_score

            + keyword_score

            + coverage_score

            + complete_score

            + descriptive_score

            + procedure_score

            + structural["boosted"]

            + structural["lexical"]

        )

        #
        # Approximate semantic contribution.
        #
        # The lexical retriever already injected
        # cosine similarity.
        #
        # Here we simply scale it according to
        # Language confidence.
        #

        semantic_score = (

            lexical_score
            *
            semantic_confidence

        )

        #
        # Final hybrid score.
        #

        final_score = (

            lexical_score
            * self.LEXICAL_WEIGHT

            +

            semantic_score
            * self.SEMANTIC_WEIGHT

            +

            structural_score
            * self.STRUCTURAL_WEIGHT

            +

            entity_score
            * self.ENTITY_WEIGHT

            +

            intent_score
            * self.INTENT_WEIGHT

            +

            keyword_score
            * self.KEYWORD_WEIGHT

            +

            coverage_score
            * self.COVERAGE_WEIGHT

            +

            complete_score
            * self.COMPLETE_MATCH_WEIGHT

            +

            descriptive_score
            * self.DESCRIPTIVE_TITLE_WEIGHT

            +

            procedure_score
            * self.PROCEDURE_TITLE_WEIGHT

        )

        return RankingBreakdown(

            lexical=lexical_score,

            structural=structural_score,

            semantic=semantic_score,

            entity=entity_score,

            intent=intent_score,

            keyword=keyword_score,

            coverage=coverage_score,

            complete=complete_score,

            descriptive=descriptive_score,

            procedure=procedure_score,

            final=final_score,

        )

    #
    # -------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------
    #

    def print_breakdown(
        self,
        match: RetrievalMatch,
        query: AnalyzedQuery,
    ) -> None:

        breakdown = self._compute_breakdown(
            match,
            query,
        )

        print()
        print("=" * 70)
        print(match.section.title)
        print("=" * 70)

        print(
            f"Lexical        : {breakdown.lexical:.3f}"
        )

        print(
            f"Semantic       : {breakdown.semantic:.3f}"
        )

        print(
            f"Structural     : {breakdown.structural:.3f}"
        )

        print(
            f"Entity         : {breakdown.entity:.3f}"
        )

        print(
            f"Intent         : {breakdown.intent:.3f}"
        )

        print(
            f"Keyword        : {breakdown.keyword:.3f}"
        )

        print(
            f"Coverage       : {breakdown.coverage:.3f}"
        )

        print(
            f"Complete Match : {breakdown.complete:.3f}"
        )

        print(
            f"Descriptive    : {breakdown.descriptive:.3f}"
        )

        print(
            f"Procedure      : {breakdown.procedure:.3f}"
        )

        print("-" * 70)

        print(
            f"FINAL          : {breakdown.final:.3f}"
        )

        print()


    #
    # -------------------------------------------------------
    # Score normalization
    # -------------------------------------------------------
    #

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

        #
        # All candidates received the same score.
        #

        if maximum == minimum:

            return [

                RetrievalMatch(
                    section=match.section,
                    score=1.0,
                )

                for match in matches

            ]

        normalized: list[
            RetrievalMatch
        ] = []

        scale = maximum - minimum

        for match in matches:

            normalized_score = (
                match.score - minimum
            ) / scale

            normalized.append(

                RetrievalMatch(

                    section=match.section,

                    score=normalized_score,

                )

            )

        return normalized

    #
    # -------------------------------------------------------
    # Utilities
    # -------------------------------------------------------
    #

    @staticmethod
    def _safe(
        value: float,
    ) -> float:

        if value < 0.0:
            return 0.0

        return value

    @staticmethod
    def _clamp(
        value: float,
        minimum: float = 0.0,
        maximum: float = 1.0,
    ) -> float:

        if value < minimum:
            return minimum

        if value > maximum:
            return maximum

        return value

    #
    # -------------------------------------------------------
    # Future hooks
    # -------------------------------------------------------
    #

    def rerank_cross_encoder(
        self,
        matches: list[RetrievalMatch],
        query: AnalyzedQuery,
    ) -> list[RetrievalMatch]:
        """
        Reserved for future CrossEncoder reranking.

        The architecture intentionally keeps this
        separated from the main hybrid ranking so that
        expensive models can be applied only to the
        best candidates.
        """

        return matches

    def rerank_rrf(
        self,
        matches: list[RetrievalMatch],
        query: AnalyzedQuery,
    ) -> list[RetrievalMatch]:
        """
        Placeholder for Reciprocal Rank Fusion.

        Keeping this method already present makes
        future experimentation much easier without
        changing the retrieval pipeline.
        """

        return matches

    #
    # -------------------------------------------------------
    # End of file
    # -------------------------------------------------------
    #
