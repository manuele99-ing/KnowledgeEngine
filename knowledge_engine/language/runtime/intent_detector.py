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
    Detects the semantic intent of a user query using
    semantic similarity.
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

    #
    # -------------------------------------------------------
    # Public API
    # -------------------------------------------------------
    #

    def detect(
        self,
        query_embedding: np.ndarray,
    ) -> Intent:

        ranking = SemanticSimilarity.rank(
            query_embedding=query_embedding,
            candidates=self._intent_embeddings,
        )

        return self._intents[
            ranking[0][0]
        ]

    def confidence(
        self,
        query_embedding: np.ndarray,
    ) -> float:

        ranking = SemanticSimilarity.rank(
            query_embedding=query_embedding,
            candidates=self._intent_embeddings,
        )

        return ranking[0][1]

    def ranking(
        self,
        query_embedding: np.ndarray,
    ) -> list[tuple[int, float]]:

        return SemanticSimilarity.rank(
            query_embedding=query_embedding,
            candidates=self._intent_embeddings,
        )

    #
    # -------------------------------------------------------
    # Debug
    # -------------------------------------------------------
    #

    def print_debug(
        self,
        query: str,
        query_embedding: np.ndarray,
    ) -> None:

        ranking = self.ranking(
            query_embedding
        )

        print()
        print("=" * 70)
        print("LANGUAGE INTENT DEBUG")
        print("=" * 70)

        print()
        print("Query")
        print(query)

        print()

        print(
            f"Embedding dimension : {query_embedding.shape[0]}"
        )

        print(
            f"Embedding norm      : {np.linalg.norm(query_embedding):.4f}"
        )

        print()

        print("Intent ranking")

        for position, (index, score) in enumerate(
            ranking,
            start=1,
        ):

            marker = ""

            if position == 1:
                marker = "<--"

            print(
                f"{position:2d}. "
                f"{self._intents[index].name:<18}"
                f"{score:.4f} {marker}"
            )

        print()

        winner = self._intents[
            ranking[0][0]
        ]

        print("Winning intent")
        print(winner.name)

        print()

        print("Confidence")
        print(f"{ranking[0][1]:.4f}")

        if len(ranking) > 1:

            margin = (
                ranking[0][1]
                - ranking[1][1]
            )

            print()

            print("Margin over second")
            print(f"{margin:.4f}")

        print("=" * 70)
        print()

    @property
    def intents(
        self,
    ) -> tuple[Intent, ...]:

        return tuple(
            self._intents
        )