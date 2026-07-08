from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from knowledge_engine.language.runtime.language_encoder import LanguageEncoder
from knowledge_engine.language.runtime.semantic_similarity import SemanticSimilarity


@dataclass(slots=True, frozen=True)
class Intent:
    """
    Represents a semantic intent.
    """

    name: str

    description: str


class IntentDetector:
    """
    Detects the semantic intent of a user query.

    The detector is based on semantic similarity rather
    than keyword matching.

    Every intent is represented by a natural language
    description encoded by the LanguageEncoder.
    """

    def __init__(
        self,
        encoder: LanguageEncoder,
        intents: list[Intent],
    ) -> None:

        self._encoder = encoder
        self._intents = intents

        #
        # Cached intent embeddings.
        #
        self._intent_embeddings = [
            encoder.encode(
                intent.description
            )
            for intent in intents
        ]

    def detect(
        self,
        query: str,
    ) -> Intent:
        """
        Returns the most semantically similar intent.
        """

        query_embedding = self._encoder.encode(
            query
        )

        index = SemanticSimilarity.most_similar(
            query_embedding,
            self._intent_embeddings,
        )

        return self._intents[index]

    def confidence(
        self,
        query: str,
    ) -> float:
        """
        Returns the similarity score of the detected intent.
        """

        query_embedding = self._encoder.encode(
            query
        )

        similarities = [
            SemanticSimilarity.cosine(
                query_embedding,
                embedding,
            )
            for embedding in self._intent_embeddings
        ]

        return max(similarities)

    @property
    def intents(
        self,
    ) -> tuple[Intent, ...]:

        return tuple(
            self._intents
        )