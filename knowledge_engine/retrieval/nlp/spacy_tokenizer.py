from __future__ import annotations

import spacy


class SpacyTokenizer:
    """
    Tokenizes and lemmatizes text using spaCy.
    """

    def __init__(
        self,
        model: str = "it_core_news_sm",
    ) -> None:
        self._nlp = spacy.load(model)

    def tokenize(
        self,
        text: str,
    ) -> tuple[str, ...]:

        doc = self._nlp(text)

        tokens = []

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

            tokens.append(lemma)

        return tuple(tokens)