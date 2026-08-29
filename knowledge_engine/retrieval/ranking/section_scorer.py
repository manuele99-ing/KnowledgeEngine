from __future__ import annotations

from knowledge_engine.retrieval.indexing.indexed_section import IndexedSection
from knowledge_engine.retrieval.query.analyzed_query import AnalyzedQuery


class SectionScorer:
    """
    Hybrid structural scorer.

    BM25 is responsible for candidate recall.

    This scorer improves precision by combining:

        â€¢ lexical evidence
        â€¢ semantic intent
        â€¢ semantic entities
        â€¢ semantic keywords
        â€¢ structural information
        â€¢ coverage

    The implementation is intentionally model and
    domain independent.

    No Disney-specific knowledge is contained here.
    """

    #
    # -------------------------------------------------------
    # CONFIGURATION
    # -------------------------------------------------------
    #

    TITLE_WEIGHT = 8.0
    PATH_WEIGHT = 5.0
    BODY_WEIGHT = 2.0

    ENTITY_TITLE_WEIGHT = 12.0
    ENTITY_PATH_WEIGHT = 8.0
    ENTITY_BODY_WEIGHT = 5.0

    COVERAGE_WEIGHT = 8.0

    BOOSTED_WEIGHT = 3.0

    SEMANTIC_KEYWORD_WEIGHT = 2.5

    INTENT_WEIGHT = 8.0

    COMPLETE_MATCH_WEIGHT = 12.0

    DESCRIPTION_TITLE_WEIGHT = 12.0

    PROCEDURE_TITLE_WEIGHT = 10.0

    #
    # Generic structural patterns.
    #
    # Completely domain independent.
    #

    INTENT_PATTERNS = {

        "procedure": (

            "step",
            "steps",
            "phase",
            "phases",
            "workflow",
            "procedure",
            "process",
            "guide",
            "guideline",
            "instruction",
            "instructions",
            "how",
            "method",
            "sequence",

            #
            # Italian
            #

            "fase",
            "fasi",
            "passaggio",
            "passaggi",
            "procedura",
            "procedimento",
            "sequenza",
            "guida",
            "istruzioni",
            "preparazione",
            "metodo",

        ),

        "description": (

            "description",
            "overview",
            "profile",
            "summary",
            "definition",
            "background",
            "characteristics",

            #
            # Italian
            #

            "descrizione",
            "panoramica",
            "profilo",
            "definizione",
            "caratteristiche",
            "introduzione",

        ),

        "definition": (

            "definition",
            "meaning",
            "overview",

            "definizione",
            "significato",
            "panoramica",

        ),

        "configuration": (

            "configuration",
            "configure",
            "settings",
            "preferences",
            "options",

            "configurazione",
            "configura",
            "impostazioni",
            "opzioni",

        ),

        "installation": (

            "installation",
            "install",
            "setup",

            "installazione",
            "installare",
            "setup",

        ),

        "location": (

            "location",
            "where",
            "path",

            "dove",
            "posizione",
            "ubicazione",
            "percorso",

        ),

        "list": (

            "list",
            "items",
            "components",
            "features",

            "lista",
            "elenco",
            "componenti",
            "funzionalitÃ ",

        ),
    }

    #
    # -------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------
    #

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

        score += self._semantic_keyword_score(
            section,
            query,
        )

        score += self._intent_score(
            section,
            query,
        )

        score += self._descriptive_title_score(
            section,
            query,
        )

        score += self._procedure_title_score(
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
    # UTILITIES
    # -------------------------------------------------------
    #

    @staticmethod
    def _lower_tokens(
        tokens: tuple[str, ...],
    ) -> set[str]:

        return {
            token.lower()
            for token in tokens
        }

    @staticmethod
    def _intent_name(
        query: AnalyzedQuery,
    ) -> str:

        #
        # Semantic intent has priority.
        #

        if query.semantic_intent:

            intent = query.semantic_intent.strip().lower()

            if intent != "unknown":
                return intent

        #
        # Fallback to lexical intent.
        #

        return query.intent.name.lower()

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

        title_tokens = self._lower_tokens(
            section.title_tokens
        )

        score = 0.0

        #
        # Lexical keywords.
        #

        for token in query.tokens:

            if token.lower() in title_tokens:
                score += self.TITLE_WEIGHT

        #
        # Semantic keywords receive a smaller boost.
        #

        for token in query.semantic_keywords:

            if token.lower() in title_tokens:
                score += self.TITLE_WEIGHT * 0.50

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

        for token in query.semantic_keywords:

            if token.lower() in path_tokens:
                score += self.PATH_WEIGHT * 0.50

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

        body_tokens = self._lower_tokens(
            section.body_tokens
        )

        score = 0.0

        for token in query.tokens:

            if token.lower() in body_tokens:
                score += self.BODY_WEIGHT

        for token in query.semantic_keywords:

            if token.lower() in body_tokens:
                score += self.BODY_WEIGHT * 0.50

        return score

    #
    # -------------------------------------------------------
    # ENTITY SCORE
    # -------------------------------------------------------
    #

    def _entity_score(
        self,
        section: IndexedSection,
        query: AnalyzedQuery,
    ) -> float:

        #
        # Prefer semantic entities.
        #

        entities = (
            query.semantic_entities
            if query.semantic_entities
            else query.entities
        )

        if not entities:
            return 0.0

        title_tokens = self._lower_tokens(
            section.title_tokens
        )

        body_tokens = self._lower_tokens(
            section.body_tokens
        )

        path_tokens = set()

        for heading in section.path:

            path_tokens.update(
                heading.lower().split()
            )

        score = 0.0

        searchable_title = " ".join(
            section.title.lower().split()
        )

        searchable_path = " ".join(
            section.path_text.lower().split()
        )

        searchable_body = " ".join(
            section.body.lower().split()
        )

        for entity in entities:

            entity = entity.lower()
            entity_parts = tuple(
                part
                for part in entity.split()
                if part
            )

            if (
                entity in searchable_title
                or
                all(
                    part in title_tokens
                    for part in entity_parts
                )
            ):

                score += self.ENTITY_TITLE_WEIGHT
                continue

            if (
                entity in searchable_path
                or
                all(
                    part in path_tokens
                    for part in entity_parts
                )
            ):

                score += self.ENTITY_PATH_WEIGHT
                continue

            if (
                entity in searchable_body
                or
                all(
                    part in body_tokens
                    for part in entity_parts
                )
            ):

                score += self.ENTITY_BODY_WEIGHT

        return score

    #
    # -------------------------------------------------------
    # DESCRIPTIVE TITLES
    # -------------------------------------------------------
    #

    def _descriptive_title_score(
        self,
        section: IndexedSection,
        query: AnalyzedQuery,
    ) -> float:

        intent = self._intent_name(
            query
        )

        if intent not in {
            "definition",
            "description",
        }:
            return 0.0

        title = section.title.lower()

        descriptive_markers = (
            "protagonista",
            "antagonista",
            "personaggio",
            "profilo",
            "chi Ã¨",
            "chi era",
            "background",
            "descrizione",
        )

        if not any(
            marker in title
            for marker in descriptive_markers
        ):
            return 0.0

        entity_terms = (
            query.semantic_entities
            if query.semantic_entities
            else query.entities
        )

        if not entity_terms:
            entity_terms = query.tokens

        for entity in entity_terms:

            parts = [
                part.lower()
                for part in entity.split()
                if part.strip()
            ]

            if parts and all(
                part in title
                for part in parts
            ):
                return self.DESCRIPTION_TITLE_WEIGHT

        return 0.0

    #
    # -------------------------------------------------------
    # PROCEDURE TITLES
    # -------------------------------------------------------
    #

    def _procedure_title_score(
        self,
        section: IndexedSection,
        query: AnalyzedQuery,
    ) -> float:

        intent = self._intent_name(
            query
        )

        if intent != "procedure":
            return 0.0

        searchable = " ".join(
            (
                section.title,
                section.path_text,
            )
        ).lower()

        procedure_markers = (
            "fase",
            "fasi",
            "step",
            "passaggio",
            "passaggi",
            "procedura",
            "processo",
            "workflow",
            "preparazione",
            "preparativi",
            "metodo",
            "sequenza",
            "istruzioni",
        )

        if any(
            marker in searchable
            for marker in procedure_markers
        ):
            return self.PROCEDURE_TITLE_WEIGHT

        return 0.0

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

        query_tokens = set()

        query_tokens.update(
            token.lower()
            for token in query.tokens
        )

        query_tokens.update(
            token.lower()
            for token in query.semantic_keywords
        )

        if not query_tokens:
            return 0.0

        section_tokens = self._lower_tokens(
            section.tokens
        )

        matched = len(
            query_tokens &
            section_tokens
        )

        if matched == 0:
            return 0.0

        coverage = (
            matched /
            len(query_tokens)
        )

        score = (
            coverage *
            self.COVERAGE_WEIGHT
        )

        #
        # Nearly complete match.
        #

        if coverage >= 0.90:

            score += 5.0

        elif coverage >= 0.75:

            score += 3.0

        elif coverage >= 0.60:

            score += 1.5

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

        section_tokens = self._lower_tokens(
            section.tokens
        )

        score = 0.0

        for token in query.boosted_tokens:

            if token.lower() in section_tokens:

                score += self.BOOSTED_WEIGHT

        return score

    #
    # -------------------------------------------------------
    # SEMANTIC KEYWORDS
    # -------------------------------------------------------
    #

    def _semantic_keyword_score(
        self,
        section: IndexedSection,
        query: AnalyzedQuery,
    ) -> float:

        if not query.semantic_keywords:
            return 0.0

        section_tokens = self._lower_tokens(
            section.tokens
        )

        score = 0.0

        matched = 0

        for keyword in query.semantic_keywords:

            keyword = keyword.lower()

            if keyword in section_tokens:

                matched += 1
                score += self.SEMANTIC_KEYWORD_WEIGHT

        #
        # Reward sections matching many semantic concepts.
        #

        if matched >= 3:

            score += 2.0

        elif matched == 2:

            score += 1.0

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

        intent = self._intent_name(
            query
        )

        patterns = self.INTENT_PATTERNS.get(
            intent
        )

        if not patterns:
            return 0.0

        #
        # Confidence produced by LanguageEngine.
        #
        # If unavailable we fallback to lexical confidence.
        #

        confidence = (
            query.semantic_confidence
            if query.semantic_confidence > 0.0
            else query.confidence
        )

        searchable = " ".join(

            (
                section.title,
                section.path_text,
                section.body,
            )

        ).lower()

        matches = 0

        for pattern in patterns:

            if pattern in searchable:

                matches += 1

        if matches == 0:
            return 0.0

        #
        # Saturating bonus.
        #
        # One match is already useful.
        # Additional matches have diminishing returns.
        #

        if matches == 1:

            factor = 0.50

        elif matches == 2:

            factor = 0.80

        elif matches == 3:

            factor = 1.00

        else:

            factor = 1.15

        return (
            self.INTENT_WEIGHT
            *
            factor
            *
            confidence
        )

        #
    # -------------------------------------------------------
    # COMPLETE MATCH
    # -------------------------------------------------------
    #

    def _complete_match_bonus(
        self,
        section: IndexedSection,
        query: AnalyzedQuery,
    ) -> float:

        query_tokens = set()

        query_tokens.update(
            token.lower()
            for token in query.tokens
        )

        query_tokens.update(
            token.lower()
            for token in query.semantic_keywords
        )

        if not query_tokens:
            return 0.0

        section_tokens = self._lower_tokens(
            section.tokens
        )

        #
        # Every query concept is present inside
        # the indexed section.
        #

        if query_tokens.issubset(
            section_tokens
        ):

            return self.COMPLETE_MATCH_WEIGHT

        return 0.0

    #
    # -------------------------------------------------------
    # DEBUG SUPPORT
    # -------------------------------------------------------
    #

    def debug_score(
        self,
        section: IndexedSection,
        query: AnalyzedQuery,
    ) -> dict[str, float]:
        """
        Returns every score component independently.

        Useful for benchmark analysis and retrieval tuning.
        """

        lexical = (
            self._title_score(section, query)
            + self._path_score(section, query)
            + self._body_score(section, query)
        )

        entity = self._entity_score(
            section,
            query,
        )

        coverage = self._coverage_score(
            section,
            query,
        )

        boosted = self._boosted_score(
            section,
            query,
        )

        semantic_keyword = (
            self._semantic_keyword_score(
                section,
                query,
            )
        )

        intent = self._intent_score(
            section,
            query,
        )

        complete = self._complete_match_bonus(
            section,
            query,
        )

        total = (
            lexical
            + entity
            + coverage
            + boosted
            + semantic_keyword
            + intent
            + complete
            + self._descriptive_title_score(
                section,
                query,
            )
            + self._procedure_title_score(
                section,
                query,
            )
        )

        return {

            "lexical": lexical,

            "entity": entity,

            "coverage": coverage,

            "boosted": boosted,

            "semantic_keyword": semantic_keyword,

            "intent": intent,

            "complete_match": complete,

            "descriptive_title": (
                self._descriptive_title_score(
                    section,
                    query,
                )
            ),

            "procedure_title": (
                self._procedure_title_score(
                    section,
                    query,
                )
            ),

            "total": total,

        }

    #
    # -------------------------------------------------------
    # OPTIONAL PRETTY DEBUG
    # -------------------------------------------------------
    #

    def print_debug(
        self,
        section: IndexedSection,
        query: AnalyzedQuery,
    ) -> None:

        values = self.debug_score(
            section,
            query,
        )

        print()
        print("=" * 60)
        print(section.title)
        print("=" * 60)

        for key, value in values.items():

            print(
                f"{key:<20}"
                f"{value:>8.3f}"
            )

        print()
