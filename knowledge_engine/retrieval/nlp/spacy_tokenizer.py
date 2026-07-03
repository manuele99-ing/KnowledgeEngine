from __future__ import annotations

import spacy

from knowledge_engine.retrieval.nlp.normalization.lemma_normalizer import (
    LemmaNormalizer,
)


class SpacyTokenizer:
    """
    Tokenizes and lemmatizes text using spaCy.

    The default model is `it_core_news_md` so that vector-based
    semantic features are available for future expansion.
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

    def tokenize(
        self,
        text: str,
    ) -> tuple[str, ...]:

        doc = self._nlp(text)

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

    def vector(
        self,
        text: str,
    ) -> tuple[float, ...]:

        doc = self._nlp(text)

        if not doc.has_vector:
            return ()

        return tuple(float(value) for value in doc.vector)

    @property
    def has_vectors(
        self,
    ) -> bool:

        return self._nlp.vocab.vectors_length > 0