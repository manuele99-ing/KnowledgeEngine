from __future__ import annotations

from knowledge_engine.domain.document import section
from knowledge_engine.retrieval.indexing.indexed_section import IndexedSection
from knowledge_engine.retrieval.query.analyzed_query import AnalyzedQuery
from knowledge_engine.retrieval.query.query_intent import QueryIntent


class SectionScorer:
    """
    Structural relevance scorer.

    BM25 is responsible for recall.

    This scorer improves precision by exploiting the
    structure already extracted during indexing.

    The implementation is intentionally generic and
    domain-independent.
    """

    TITLE_WEIGHT = 8.0
    PATH_WEIGHT = 5.0
    BODY_WEIGHT = 2.0

    ENTITY_TITLE_WEIGHT = 10.0
    ENTITY_BODY_WEIGHT = 5.0

    COVERAGE_WEIGHT = 8.0

    BOOSTED_WEIGHT = 3.0

    INTENT_WEIGHT = 6.0

    COMPLETE_MATCH_WEIGHT = 12.0

    def score(
        self,
        section: IndexedSection,
        query: AnalyzedQuery,
    ) -> float:

        score = 0.0

        score += self._title_score(
            section,
            query,
        )

        score += self._path_score(
            section,
            query,
        )

        score += self._body_score(
            section,
            query,
        )

        score += self._entity_score(
            section,
            query,
        )

        score += self._coverage_score(
            section,
            query,
        )

        score += self._boosted_score(
            section,
            query,
        )

        score += self._intent_score(
            section,
            query,
        )

        score += self._complete_match_bonus(
            section,
            query,
        )

        return score

    #
    # -------------------------------------------------------
    # TITLE
    # -------------------------------------------------------
    #

    def _title_score(
        self,
        section: IndexedSection,
        query: AnalyzedQuery,
    ) -> float:

        title_tokens = {
            token.lower()
            for token in section.title_tokens
        }

        score = 0.0

        for token in query.tokens:

            if token.lower() in title_tokens:
                score += self.TITLE_WEIGHT

        return score

    #
    # -------------------------------------------------------
    # PATH
    # -------------------------------------------------------
    #

    def _path_score(
        self,
        section: IndexedSection,
        query: AnalyzedQuery,
    ) -> float:

        path_tokens = set()

        for heading in section.path:
            path_tokens.update(
                heading.lower().split()
            )

        score = 0.0

        for token in query.tokens:

            if token.lower() in path_tokens:
                score += self.PATH_WEIGHT

        return score

    #
    # -------------------------------------------------------
    # BODY
    # -------------------------------------------------------
    #

    def _body_score(
        self,
        section: IndexedSection,
        query: AnalyzedQuery,
    ) -> float:

        tokens = {
            token.lower()
            for token in section.body_tokens
        }

        score = 0.0

        for token in query.tokens:

            if token.lower() in tokens:
                score += self.BODY_WEIGHT

        return score
    

        #
    # -------------------------------------------------------
    # ENTITY
    # -------------------------------------------------------
    #

    def _entity_score(
        self,
        section: IndexedSection,
        query: AnalyzedQuery,
    ) -> float:

        if not query.entities:
            return 0.0

        title_tokens = {
            token.lower()
            for token in section.title_tokens
        }

        tokens = {
            token.lower()
            for token in section.body_tokens
        }

        score = 0.0

        for entity in query.entities:

            entity = entity.lower()

            if entity in title_tokens:
                score += self.ENTITY_TITLE_WEIGHT

            elif entity in tokens:
                score += self.ENTITY_BODY_WEIGHT

        return score

    #
    # -------------------------------------------------------
    # COVERAGE
    # -------------------------------------------------------
    #

    def _coverage_score(
        self,
        section: IndexedSection,
        query: AnalyzedQuery,
    ) -> float:

        query_tokens = set(query.all_tokens)

        if not query_tokens:
            return 0.0

        section_tokens = {
            token.lower()
            for token in section.tokens
        }

        matched = len(
            query_tokens &
            section_tokens
        )

        if matched == 0:
            return 0.0

        coverage = matched / len(query_tokens)

        score = coverage * self.COVERAGE_WEIGHT

        #
        # Reward sections matching almost the
        # entire query.
        #

        if coverage >= 0.80:
            score += 4.0

        elif coverage >= 0.60:
            score += 2.0

        return score

    #
    # -------------------------------------------------------
    # BOOSTED TOKENS
    # -------------------------------------------------------
    #

    def _boosted_score(
        self,
        section: IndexedSection,
        query: AnalyzedQuery,
    ) -> float:

        if not query.boosted_tokens:
            return 0.0

        tokens = {
            token.lower()
            for token in section.body_tokens
        }

        score = 0.0

        for token in query.boosted_tokens:

            if token.lower() in tokens:
                score += self.BOOSTED_WEIGHT

        return score

    #
    # -------------------------------------------------------
    # INTENT
    # -------------------------------------------------------
    #

    def _intent_score(
        self,
        section: IndexedSection,
        query: AnalyzedQuery,
    ) -> float:

        title = section.title.lower()

        path = " ".join(section.path).lower()

        score = 0.0

        if query.intent == QueryIntent.PROCEDURE:

            for keyword in (
                "procedura",
                "procedimento",
                "passaggi",
                "guida",
                "istruzioni",
            ):
                if keyword in title or keyword in path:
                    score += self.INTENT_WEIGHT

        elif query.intent == QueryIntent.INSTALLATION:

            for keyword in (
                "installazione",
                "installare",
                "montaggio",
                "setup",
            ):
                if keyword in title or keyword in path:
                    score += self.INTENT_WEIGHT

        elif query.intent == QueryIntent.CONFIGURATION:

            for keyword in (
                "configurazione",
                "configura",
                "impostazioni",
                "setup",
            ):
                if keyword in title or keyword in path:
                    score += self.INTENT_WEIGHT

        elif query.intent == QueryIntent.DEFINITION:

            for keyword in (
                "definizione",
                "descrizione",
                "introduzione",
                "panoramica",
                "overview",
            ):
                if keyword in title or keyword in path:
                    score += self.INTENT_WEIGHT

        elif query.intent == QueryIntent.DESCRIPTION:

            for keyword in (
                "descrizione",
                "caratteristiche",
                "funzionamento",
                "panoramica",
            ):
                if keyword in title or keyword in path:
                    score += self.INTENT_WEIGHT

        elif query.intent == QueryIntent.LOCATION:

            for keyword in (
                "posizione",
                "ubicazione",
                "dove",
                "collocazione",
            ):
                if keyword in title or keyword in path:
                    score += self.INTENT_WEIGHT

        elif query.intent == QueryIntent.LIST:

            for keyword in (
                "elenco",
                "lista",
                "componenti",
                "funzioni",
                "funzionalità",
            ):
                if keyword in title or keyword in path:
                    score += self.INTENT_WEIGHT

        return score

    #
    # -------------------------------------------------------
    # COMPLETE QUERY MATCH
    # -------------------------------------------------------
    #

    def _complete_match_bonus(
        self,
        section: IndexedSection,
        query: AnalyzedQuery,
    ) -> float:

        query_tokens = set(query.tokens)

        if not query_tokens:
            return 0.0

        section_tokens = {
            token.lower()
            for token in section.tokens
        }

        #
        # Every query token is present inside the section.
        #

        if query_tokens.issubset(section_tokens):
            return self.COMPLETE_MATCH_WEIGHT

        return 0.0