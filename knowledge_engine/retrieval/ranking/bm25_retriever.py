from __future__ import annotations

from math import sqrt

from rank_bm25 import BM25Okapi

from knowledge_engine.retrieval.indexing.section_index import SectionIndex
from knowledge_engine.retrieval.query.analyzed_query import AnalyzedQuery
from knowledge_engine.retrieval.query.query_processor import QueryProcessor
from knowledge_engine.retrieval.ranking.smart_section_ranker import (
    SmartSectionRanker,
)
from knowledge_engine.retrieval.retrieval_match import RetrievalMatch

class BM25Retriever:
    """
    BM25 lexical retriever.

    Responsibilities

    - lexical retrieval
    - candidate generation
    - semantic scoring
    - delegate structural ranking

    The retriever is intentionally independent from
    spaCy and linguistic analysis.

    Every query is processed through QueryProcessor.
    """

    def __init__(
        self,
        section_index: SectionIndex,
        query_processor: QueryProcessor,
        k1: float = 1.5,
        b: float = 0.0,
        semantic_weight: float = 0.50,
    ) -> None:

        self._section_index = section_index

        self._query_processor = query_processor

        self._semantic_weight = semantic_weight

        self._ranker = SmartSectionRanker()

        corpus = [
            list(section.tokens)
            for section in section_index.sections
        ]

        self._bm25 = BM25Okapi(
            corpus,
            k1=k1,
            b=b,
        )

    #
    # -------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------
    #

    def search(
        self,
        question: str,
        top_k: int = 5,
    ) -> tuple[list[RetrievalMatch], list[str]]:

        analyzed_query = (
            self._query_processor.process(
                question
            )
        )

        matches = self._retrieve(
            analyzed_query
        )

        if not matches:

            return (
                [],
                list(analyzed_query.all_tokens),
            )

        matches = self._rank(
            matches,
            analyzed_query,
        )

        matches = self._filter(
            matches
        )

        #
        # Ordine finale.
        #

        matches = sorted(
            matches,
            key=lambda item: item.score,
            reverse=True,
        )

        return (
            matches[:top_k],
            list(analyzed_query.all_tokens),
        )

    #
    # -------------------------------------------------------
    # BM25 Retrieval
    # -------------------------------------------------------
    #

    def _retrieve(
        self,
        query: AnalyzedQuery,
    ) -> list[RetrievalMatch]:

        #
        # Costruiamo una query pesata.
        #

        weighted_query: list[str] = []

        weighted_query.extend(query.tokens)

        #
        # Le entità sono molto importanti.
        #

        for token in query.entities:
            weighted_query.extend([token.lower()] * 3)

        #
        # I boosted token hanno peso doppio.
        #

        for token in query.boosted_tokens:
            weighted_query.extend([token.lower()] * 2)

        #
        # I token espansi migliorano il recall.
        #

        for token in query.expanded_tokens:

            token = token.lower()

            if token not in weighted_query:
                weighted_query.append(token)

        scores = self._bm25.get_scores(
            weighted_query
        )

        query_vector = query.semantic_embedding

        matches: list[RetrievalMatch] = []

        for index, bm25_score in enumerate(scores):

            section = self._section_index.sections[index]

            semantic_score = 0.0

            if (
                query_vector is not None
                and
                section.embedding is not None
            ):
                semantic_score = self._cosine_similarity(
                    query_vector,
                    section.embedding,
                )

            final_score = (
                bm25_score
                + semantic_score * self._semantic_weight
            )

            #
            # Bonus se i token principali sono
            # presenti nel titolo della sezione.
            #

            title_tokens = {
                token.lower()
                for token in section.title_tokens
            }

            title_matches = sum(
                1
                for token in query.tokens
                if token.lower() in title_tokens
            )

            final_score += title_matches * 0.75

            if final_score <= 0:
                continue

            matches.append(
                RetrievalMatch(
                    section=section,
                    score=float(final_score),
                )
            )

        return matches
    
    #
    # -------------------------------------------------------
    # Structural Ranking
    # -------------------------------------------------------
    #

    def _rank(
        self,
        matches: list[RetrievalMatch],
        query: AnalyzedQuery,
    ) -> list[RetrievalMatch]:
        """
        Second-stage ranking.

        BM25 maximizes recall.

        SmartSectionRanker improves precision using
        structural information extracted during indexing.
        """

        candidates = sorted(
            matches,
            key=lambda match: match.score,
            reverse=True,
        )

        #
        # Only rerank the best lexical candidates.
        #

        candidates = candidates[:40]

        ranked = self._ranker.rerank(
            matches=candidates,
            query=query,
        )

        #
        # Normalize scores to [0, 1].
        #

        ranked = self._ranker.normalize(
            ranked
        )

        return ranked

    #
    # -------------------------------------------------------
    # Candidate Filtering
    # -------------------------------------------------------
    #

    def _filter(
        self,
        matches: list[RetrievalMatch],
    ) -> list[RetrievalMatch]:

        if not matches:
            return []

        best_score = matches[0].score

        #
        # Nessun candidato realmente valido.
        #

        if best_score <= 0.05:
            return []

        filtered: list[RetrievalMatch] = []

        #
        # Manteniamo tutti i risultati vicini
        # al migliore.
        #

        dynamic_threshold = best_score * 0.35

        #
        # Evitiamo di eliminare troppo presto
        # sezioni che il reranker ha promosso.
        #

        dynamic_threshold = max(
            dynamic_threshold,
            0.20,
        )

        for match in matches:

            if match.score >= dynamic_threshold:
                filtered.append(match)

        #
        # Garantiamo almeno 5 candidati.
        #

        if len(filtered) < 5:

            filtered = matches[
                : min(
                    5,
                    len(matches),
                )
            ]

        return filtered

    #
    # -------------------------------------------------------
    # Utilities
    # -------------------------------------------------------
    #

    @staticmethod
    def _cosine_similarity(
        vector_a,
        vector_b,
    ) -> float:

        if vector_a is None:
            return 0.0

        if vector_b is None:
            return 0.0

        dot = sum(
            a * b
            for a, b in zip(
                vector_a,
                vector_b,
            )
        )

        norm_a = sqrt(
            sum(
                value * value
                for value in vector_a
            )
        )

        norm_b = sqrt(
            sum(
                value * value
                for value in vector_b
            )
        )

        if norm_a == 0:
            return 0.0

        if norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)


    #
    # -------------------------------------------------------
    # Confidence Estimation
    # -------------------------------------------------------
    #

    def _confidence(
        self,
        matches: list[RetrievalMatch],
    ) -> float:
        """
        Estimates retrieval confidence from the score
        distribution.

        The confidence is independent from BM25 and can
        later be reused by Hybrid or FAISS retrievers.
        """

        if not matches:
            return 0.0

        if len(matches) == 1:
            return 1.0

        best = matches[0].score
        second = matches[1].score

        if best <= 0:
            return 0.0

        confidence = (
            best - second
        ) / best

        return max(
            0.0,
            min(
                confidence,
                1.0,
            ),
        )

    #
    # -------------------------------------------------------
    # Match Reduction
    # -------------------------------------------------------
    #

    def _reduce(
        self,
        matches: list[RetrievalMatch],
        top_k: int,
    ) -> list[RetrievalMatch]:
        """
        Reduces the candidate list while preserving
        score ordering.
        """

        if len(matches) <= top_k:
            return matches

        return matches[:top_k]

    #
    # -------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------
    #

    @staticmethod
    def _sort_matches(
        matches: list[RetrievalMatch],
    ) -> list[RetrievalMatch]:

        return sorted(
            matches,
            key=lambda match: match.score,
            reverse=True,
        )

    @staticmethod
    def _has_results(
        matches: list[RetrievalMatch],
    ) -> bool:

        return len(matches) > 0

    @staticmethod
    def _query_terms(
        query: AnalyzedQuery,
    ) -> list[str]:
        """
        Returns the lexical representation of the query
        used by BM25.
        """

        return list(query.all_tokens)

    @staticmethod
    def _candidate_count(
        matches: list[RetrievalMatch],
    ) -> int:

        return len(matches)
    


        #
    # -------------------------------------------------------
    # Confidence Estimation
    # -------------------------------------------------------
    #

    def _confidence(
        self,
        matches: list[RetrievalMatch],
    ) -> float:
        """
        Estimates retrieval confidence from the score
        distribution.

        The confidence is independent from BM25 and can
        later be reused by Hybrid or FAISS retrievers.
        """

        if not matches:
            return 0.0

        if len(matches) == 1:
            return 1.0

        best = matches[0].score
        second = matches[1].score

        if best <= 0:
            return 0.0

        confidence = (
            best - second
        ) / best

        return max(
            0.0,
            min(
                confidence,
                1.0,
            ),
        )

    #
    # -------------------------------------------------------
    # Match Reduction
    # -------------------------------------------------------
    #

    def _reduce(
        self,
        matches: list[RetrievalMatch],
        top_k: int,
    ) -> list[RetrievalMatch]:
        """
        Reduces the candidate list while preserving
        score ordering.
        """

        if len(matches) <= top_k:
            return matches

        return matches[:top_k]

    #
    # -------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------
    #

    @staticmethod
    def _sort_matches(
        matches: list[RetrievalMatch],
    ) -> list[RetrievalMatch]:

        return sorted(
            matches,
            key=lambda match: match.score,
            reverse=True,
        )

    @staticmethod
    def _has_results(
        matches: list[RetrievalMatch],
    ) -> bool:

        return len(matches) > 0

    @staticmethod
    def _query_terms(
        query: AnalyzedQuery,
    ) -> list[str]:
        """
        Returns the lexical representation of the query
        used by BM25.
        """

        return list(query.all_tokens)

    @staticmethod
    def _candidate_count(
        matches: list[RetrievalMatch],
    ) -> int:

        return len(matches)

