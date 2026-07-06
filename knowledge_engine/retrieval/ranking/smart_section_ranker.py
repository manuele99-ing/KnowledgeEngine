from __future__ import annotations

from knowledge_engine.retrieval.query.analyzed_query import AnalyzedQuery
from knowledge_engine.retrieval.retrieval_match import RetrievalMatch


class SmartSectionRanker:
    """
    Structural reranker.

    BM25 is responsible for recall.

    This component improves precision by exploiting
    the structure of indexed sections instead of using
    domain-specific rules.

    The ranker is intentionally independent from
    BM25, embeddings and LLMs.
    """

    #
    # Feature weights.
    #

    TITLE_WEIGHT = 3.00

    PATH_WEIGHT = 1.50

    BODY_WEIGHT = 1.00

    ENTITY_WEIGHT = 2.50

    BOOSTED_WEIGHT = 2.00

    COVERAGE_WEIGHT = 4.00

    def rerank(
        self,
        matches: list[RetrievalMatch],
        query: AnalyzedQuery,
    ) -> list[RetrievalMatch]:

        reranked: list[tuple[float, RetrievalMatch]] = []

        for match in matches:

            score = match.score

            section = match.section

            score += (
                self._title_score(
                    section,
                    query,
                )
            )

            score += (
                self._path_score(
                    section,
                    query,
                )
            )

            score += (
                self._body_score(
                    section,
                    query,
                )
            )

            score += (
                self._entity_score(
                    section,
                    query,
                )
            )

            score += (
                self._coverage_score(
                    section,
                    query,
                )
            )

            reranked.append(
                (
                    score,
                    RetrievalMatch(
                        section=section,
                        score=score,
                    ),
                )
            )

        reranked.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            match
            for _, match in reranked
        ]

    #
    # --------------------------------------------------
    # TITLE
    # --------------------------------------------------
    #

    def _title_score(
        self,
        section,
        query: AnalyzedQuery,
    ) -> float:

        overlap = (
            set(query.all_tokens)
            &
            set(section.title_tokens)
        )

        return (
            len(overlap)
            * self.TITLE_WEIGHT
        )

    #
    # --------------------------------------------------
    # PATH
    # --------------------------------------------------
    #

    def _path_score(
        self,
        section,
        query: AnalyzedQuery,
    ) -> float:

        path_tokens = set()

        for heading in section.path:

            for token in heading.lower().split():

                path_tokens.add(token)

        overlap = (
            set(query.all_tokens)
            &
            path_tokens
        )

        return (
            len(overlap)
            * self.PATH_WEIGHT
        )

    #
    # --------------------------------------------------
    # BODY
    # --------------------------------------------------
    #

    def _body_score(
        self,
        section,
        query: AnalyzedQuery,
    ) -> float:

        overlap = (
            set(query.tokens)
            &
            set(section.body_tokens)
        )

        return (
            len(overlap)
            * self.BODY_WEIGHT
        )
    
    #
    # --------------------------------------------------
    # ENTITY
    # --------------------------------------------------
    #

    def _entity_score(
        self,
        section,
        query: AnalyzedQuery,
    ) -> float:

        if not query.entities:
            return 0.0

        title_tokens = {
            token.lower()
            for token in section.title_tokens
        }

        body_tokens = {
            token.lower()
            for token in section.body_tokens
        }

        score = 0.0

        for entity in query.entities:

            entity = entity.lower()

            #
            # Exact title match.
            #

            if entity in title_tokens:
                score += self.ENTITY_WEIGHT

            #
            # Exact body match.
            #

            elif entity in body_tokens:
                score += self.ENTITY_WEIGHT * 0.50

        return score

    #
    # --------------------------------------------------
    # COVERAGE
    # --------------------------------------------------
    #

    def _coverage_score(
        self,
        section,
        query: AnalyzedQuery,
    ) -> float:

        query_tokens = set(query.all_tokens)

        if not query_tokens:
            return 0.0

        section_tokens = set(section.tokens)

        matched = len(
            query_tokens &
            section_tokens
        )

        coverage = matched / len(query_tokens)

        return coverage * self.COVERAGE_WEIGHT

    #
    # --------------------------------------------------
    # OPTIONAL BOOST
    # --------------------------------------------------
    #

    def boosted_score(
        self,
        section,
        query: AnalyzedQuery,
    ) -> float:
        """
        Generic boost using the tokens selected
        by the QueryAnalyzer.

        No domain-specific assumptions are made.
        """

        if not query.boosted_tokens:
            return 0.0

        overlap = (
            set(query.boosted_tokens)
            &
            set(section.tokens)
        )

        return (
            len(overlap)
            * self.BOOSTED_WEIGHT
        )

    #
    # --------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------
    #

    @staticmethod
    def normalize(
        matches: list[RetrievalMatch],
    ) -> list[RetrievalMatch]:

        if not matches:
            return matches

        maximum = max(
            match.score
            for match in matches
        )

        if maximum <= 0:
            return matches

        normalized: list[
            RetrievalMatch
        ] = []

        for match in matches:

            normalized.append(
                RetrievalMatch(
                    section=match.section,
                    score=match.score / maximum,
                )
            )

        return normalized

