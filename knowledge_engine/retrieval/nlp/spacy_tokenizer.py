from __future__ import annotations

import spacy
from spacy.language import Language
from spacy.tokens import Doc

from knowledge_engine.retrieval.nlp.nlp_document import NLPDocument
from knowledge_engine.retrieval.nlp.normalization.lemma_normalizer import (
    LemmaNormalizer,
)


class SpacyTokenizer:
    """
    Central NLP service for the retrieval pipeline.

    Responsibilities

    - parse text only once
    - tokenize
    - lemmatize
    - normalize lemmas
    - expose vectors
    - expose the parsed spaCy document

    Every retrieval component should reuse the same parsed
    document instead of calling spaCy multiple times.
    """

    def __init__(
        self,
        model: str = "it_core_news_md",
    ) -> None:

        try:
            self._nlp = spacy.load(model)
        except OSError:
            self._nlp = spacy.load("it_core_news_sm")

        self._normalizer = LemmaNormalizer()

    @property
    def language(
        self,
    ) -> Language:
        return self._nlp

    @property
    def has_vectors(
        self,
    ) -> bool:

        return self._nlp.vocab.vectors_length > 0

    def parse(
        self,
        text: str,
    ) -> NLPDocument:
        """
        Parses text once and returns every linguistic
        representation needed by the retrieval pipeline.
        """

        doc = self._nlp(text)

        tokens = self._extract_tokens(doc)

        vector: tuple[float, ...]

        if doc.has_vector:
            vector = tuple(float(v) for v in doc.vector)
        else:
            vector = ()

        return NLPDocument(
            doc=doc,
            tokens=tokens,
            vector=vector,
        )

    def tokenize(
        self,
        text: str,
    ) -> tuple[str, ...]:
        """
        Compatibility wrapper.

        Existing code can continue using tokenize()
        while new components should prefer parse().
        """

        return self.parse(text).tokens

    def vector(
        self,
        text: str,
    ) -> tuple[float, ...]:
        """
        Compatibility wrapper.
        """

        return self.parse(text).vector

    def _extract_tokens(
        self,
        doc: Doc,
    ) -> tuple[str, ...]:

        tokens: list[str] = []

        for token in doc:

            if token.is_space:
                continue

            if token.is_punct:
                continue

            if token.is_stop:
                continue

            lemma = token.lemma_.strip().lower()

            if not lemma:
                continue

            lemma = self._normalizer.normalize(lemma)

            tokens.append(lemma)

        return tuple(tokens)