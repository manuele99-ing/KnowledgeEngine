from __future__ import annotations

from knowledge_engine.retrieval.nlp.spacy_tokenizer import SpacyTokenizer
from knowledge_engine.retrieval.query.analyzed_query import AnalyzedQuery
from knowledge_engine.retrieval.query.query_analyzer import QueryAnalyzer


class QueryProcessor:
    """
    High-level entry point for query processing.

    The processor centralizes every NLP operation so that
    downstream retrieval components remain completely
    independent from spaCy.
    """

    def __init__(
        self,
        tokenizer: SpacyTokenizer | None = None,
        analyzer: QueryAnalyzer | None = None,
    ) -> None:

        self._tokenizer = tokenizer or SpacyTokenizer()
        if analyzer is None:
            raise ValueError(
                "QueryProcessor requires a QueryAnalyzer instance."
            )

        self._analyzer = analyzer
        
    def process(
        self,
        question: str,
    ) -> AnalyzedQuery:

        parsed = self._tokenizer.parse(question)

        return self._analyzer.analyze(
            document=parsed,
            original_text=question,
        )

    def parse(
        self,
        question: str,
    ):
        """
        Exposes the parsed NLP document for future
        semantic retrieval modules.
        """

        return self._tokenizer.parse(question)

    @property
    def tokenizer(
        self,
    ) -> SpacyTokenizer:

        return self._tokenizer