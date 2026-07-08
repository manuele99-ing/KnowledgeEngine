from __future__ import annotations

from knowledge_engine.language.runtime.language_encoder import LanguageEncoder
from knowledge_engine.language.runtime.semantic_query import SemanticQuery
from knowledge_engine.language.runtime.intent_detector import (
    Intent,
    IntentDetector,
)


class SemanticAnalyzer:
    """
    High level semantic analyzer.

    This class orchestrates every semantic component of the
    Language module.

    It is intentionally independent from Retrieval.

    Responsibilities

        • generate semantic embeddings

        • detect semantic intent

        • extract semantic keywords
          (future implementation)

        • detect semantic entities
          (future implementation)

        • build a SemanticQuery
    """

    def __init__(
        self,
        encoder: LanguageEncoder,
        intent_detector: IntentDetector,
    ) -> None:

        self._encoder = encoder
        self._intent_detector = intent_detector

    def analyze(
        self,
        query: str,
    ) -> SemanticQuery:
        """
        Produces a complete semantic representation
        of a user query.
        """

        embedding = self._encode(query)

        intent = self._detect_intent(query)

        confidence = self._intent_confidence(query)

        #
        # Placeholder.
        #
        # These components will later be replaced by
        # semantic extractors trained with the language
        # model.
        #

        keywords = self._extract_keywords(query)

        entities = self._extract_entities(query)

        expanded = self._expand_keywords(
            keywords
        )

        return SemanticQuery(
            text=query,
            embedding=embedding,
            intent=intent.name.lower(),
            keywords=keywords,
            entities=entities,
            expanded_keywords=expanded,
            confidence=confidence,
        )

    def _encode(
        self,
        query: str,
    ):

        try:

            return self._encoder.encode(query)

        except NotImplementedError:

            #
            # Encoder not available yet.
            #

            return None

    def _detect_intent(
        self,
        query: str,
    ) -> Intent:

        return self._intent_detector.detect(
            query
        )

    def _intent_confidence(
        self,
        query: str,
    ) -> float:

        return self._intent_detector.confidence(
            query
        )

    #
    # ------------------------------------------------------------------
    # Future semantic extractors
    # ------------------------------------------------------------------
    #

    def _extract_keywords(
        self,
        query: str,
    ) -> tuple[str, ...]:
        """
        Temporary keyword extraction.

        This implementation is intentionally simple.

        It will later be replaced by a semantic keyword
        extractor based on the language model.
        """

        words = []

        seen = set()

        for word in query.lower().split():

            word = (
                word.replace(",", "")
                .replace(".", "")
                .replace("?", "")
                .replace("!", "")
                .strip()
            )

            if len(word) < 3:
                continue

            if word in seen:
                continue

            seen.add(word)

            words.append(word)

        return tuple(words)

    def _extract_entities(
        self,
        query: str,
    ) -> tuple[str, ...]:
        """
        Placeholder.

        Future implementation:

            semantic NER
            entity linking
            ontology lookup
        """

        return ()

    def _expand_keywords(
        self,
        keywords: tuple[str, ...],
    ) -> tuple[str, ...]:
        """
        Placeholder.

        Future implementation:

            synonyms

            semantic expansion

            multilingual expansion

            ontology expansion
        """

        return ()