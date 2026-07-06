from __future__ import annotations

from dataclasses import dataclass, field

from knowledge_engine.retrieval.query.query_intent import QueryIntent


@dataclass(frozen=True, slots=True)
class AnalyzedQuery:
    """
    Structured representation of a user query.

    The QueryAnalyzer transforms a natural language
    question into this object so that downstream
    retrieval components can reason on the user's
    intent instead of raw text alone.
    """

    #
    # Original question
    #

    text: str

    #
    # Detected intent
    #

    intent: QueryIntent = QueryIntent.UNKNOWN

    #
    # Tokenized query
    #

    tokens: tuple[str, ...] = ()

    #
    # Named entities extracted from the query.
    #

    entities: tuple[str, ...] = ()

    #
    # Tokens that should receive additional
    # weight during retrieval.
    #

    boosted_tokens: tuple[str, ...] = ()

    #
    # Expanded tokens (synonyms, normalized
    # forms, semantic equivalents).
    #

    expanded_tokens: tuple[str, ...] = ()

    #
    # Confidence of the detected intent.
    #

    confidence: float = 0.0

    @property
    def all_tokens(
        self,
    ) -> tuple[str, ...]:
        """
        Returns every token that should participate
        in retrieval.
        """

        seen: set[str] = set()
        result: list[str] = []

        for collection in (
            self.tokens,
            self.entities,
            self.boosted_tokens,
            self.expanded_tokens,
        ):
            for token in collection:
                if token in seen:
                    continue

                seen.add(token)
                result.append(token)

        return tuple(result)