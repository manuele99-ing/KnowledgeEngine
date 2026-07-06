from __future__ import annotations

from knowledge_engine.retrieval.retrieval_match import RetrievalMatch


class SmartSectionRanker:
    """
    Second-stage reranker.

    BM25 is responsible for recall.
    This component improves precision by considering the
    structure of each indexed section.

    No LLM or embeddings are required.
    """

    _DEFINITION_WORDS = {
        "chi",
        "descrivi",
        "descrizione",
        "parlami",
        "parlare",
        "ruolo",
        "personaggio",
        "protagonista",
    }

    def rerank(
        self,
        matches: list[RetrievalMatch],
        query_tokens: list[str],
    ) -> list[RetrievalMatch]:

        reranked: list[tuple[float, RetrievalMatch]] = []

        definition_query = any(
            token in self._DEFINITION_WORDS
            for token in query_tokens
        )

        for match in matches:

            section = match.section

            score = match.score

            # ---------------------------------------
            # Title bonus
            # ---------------------------------------

            title_overlap = len(
                set(query_tokens)
                &
                set(section.title_tokens)
            )

            score += title_overlap * 2.50

            # ---------------------------------------
            # Path bonus
            # ---------------------------------------

            path_overlap = len(
                set(query_tokens)
                &
                set(section.tokens)
            )

            score += path_overlap * 0.35

            # ---------------------------------------
            # Definition bonus
            # ---------------------------------------

            if definition_query:

                title = section.title.lower()

                keywords = (
                    "protagonista",
                    "principessa",
                    "assistente",
                    "saggio",
                    "antagonista",
                    "personaggio",
                )

                if any(word in title for word in keywords):
                    score += 4.0

            # ---------------------------------------
            # Exact title match
            # ---------------------------------------

            for token in query_tokens:

                if token in section.title_tokens:
                    score += 3.0

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