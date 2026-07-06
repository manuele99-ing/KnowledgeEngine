from __future__ import annotations

from knowledge_engine.retrieval.nlp.nlp_document import NLPDocument
from knowledge_engine.retrieval.query.analyzed_query import AnalyzedQuery
from knowledge_engine.retrieval.query.entity_extractor import EntityExtractor
from knowledge_engine.retrieval.query.intent_detector import IntentDetector
from knowledge_engine.retrieval.query.keyword_extractor import KeywordExtractor
from knowledge_engine.retrieval.query.query_expander import QueryExpander


class QueryAnalyzer:
    """
    High-level query analysis pipeline.

    The analyzer never invokes spaCy directly.
    It receives an already parsed NLPDocument so that
    the sentence is parsed only once inside the retrieval
    pipeline.
    """

    def __init__(
        self,
    ) -> None:

        self._intent_detector = IntentDetector()
        self._keyword_extractor = KeywordExtractor()
        self._entity_extractor = EntityExtractor()
        self._query_expander = QueryExpander()

    def analyze(
        self,
        document: NLPDocument,
        original_text: str,
    ) -> AnalyzedQuery:

        doc = document.doc

        intent = self._intent_detector.detect(doc)

        keywords = self._keyword_extractor.extract(doc)

        entities = self._entity_extractor.extract(doc)

        expanded = self._query_expander.expand(doc)

        boosted = self._build_boosted_tokens(
            keywords,
            entities,
        )

        confidence = self._estimate_confidence(
            keywords,
            entities,
            intent,
        )

        return AnalyzedQuery(
            text=original_text,
            intent=intent,
            tokens=keywords,
            entities=entities,
            boosted_tokens=boosted,
            expanded_tokens=expanded,
            confidence=confidence,
        )

    @staticmethod
    def _build_boosted_tokens(
        keywords: tuple[str, ...],
        entities: tuple[str, ...],
    ) -> tuple[str, ...]:

        boosted: list[str] = []
        seen: set[str] = set()

        #
        # Named entities receive the highest priority.
        #

        for entity in entities:

            entity = entity.lower()

            if entity in seen:
                continue

            seen.add(entity)
            boosted.append(entity)

        #
        # Remaining keywords.
        #

        for keyword in keywords:

            if keyword in seen:
                continue

            seen.add(keyword)
            boosted.append(keyword)

        return tuple(boosted)

    @staticmethod
    def _estimate_confidence(
        keywords: tuple[str, ...],
        entities: tuple[str, ...],
        intent,
    ) -> float:

        score = 0.0

        if keywords:
            score += 0.40

        if entities:
            score += 0.35

        if intent.name != "UNKNOWN":
            score += 0.25

        return min(score, 1.0)