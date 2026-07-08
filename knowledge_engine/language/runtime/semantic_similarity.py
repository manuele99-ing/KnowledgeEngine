from __future__ import annotations

import numpy as np


class SemanticSimilarity:
    """
    Utility class for comparing semantic embeddings.

    This class is intentionally stateless and contains only
    mathematical operations.

    Future implementations may expose additional metrics,
    but cosine similarity will remain the default.
    """

    @staticmethod
    def cosine(
        embedding_a: np.ndarray,
        embedding_b: np.ndarray,
    ) -> float:
        """
        Computes the cosine similarity between two embeddings.
        """

        norm_a = np.linalg.norm(embedding_a)
        norm_b = np.linalg.norm(embedding_b)

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return float(
            np.dot(embedding_a, embedding_b)
            / (norm_a * norm_b)
        )

    @staticmethod
    def distance(
        embedding_a: np.ndarray,
        embedding_b: np.ndarray,
    ) -> float:
        """
        Computes the cosine distance.

        Distance = 1 - cosine similarity.
        """

        return (
            1.0
            - SemanticSimilarity.cosine(
                embedding_a,
                embedding_b,
            )
        )

    @staticmethod
    def most_similar(
        query_embedding: np.ndarray,
        candidates: list[np.ndarray],
    ) -> int:
        """
        Returns the index of the most similar embedding.
        """

        if not candidates:
            raise ValueError(
                "Candidate list is empty."
            )

        best_index = 0
        best_score = -1.0

        for index, embedding in enumerate(candidates):

            score = SemanticSimilarity.cosine(
                query_embedding,
                embedding,
            )

            if score > best_score:

                best_score = score
                best_index = index

        return best_index

    @staticmethod
    def rank(
        query_embedding: np.ndarray,
        candidates: list[np.ndarray],
    ) -> list[tuple[int, float]]:
        """
        Returns all candidates ordered by semantic similarity.

        Returns
        -------
        list[(index, similarity)]
        """

        ranking = []

        for index, embedding in enumerate(candidates):

            similarity = SemanticSimilarity.cosine(
                query_embedding,
                embedding,
            )

            ranking.append(
                (
                    index,
                    similarity,
                )
            )

        ranking.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return ranking