from __future__ import annotations

from typing import Iterable

from spacy.language import Language


class SentenceSplitter:
    """
    Splits documents into linguistic sentences using spaCy.

    The splitter is intentionally language-agnostic.

    The spaCy pipeline is injected from outside so the
    caller decides which language model to use.

    The splitter never performs:

        - cleaning
        - duplicate removal
        - language detection

    It only converts a document into individual sentences.
    """

    def __init__(
        self,
        nlp: Language,
    ) -> None:

        self._nlp = nlp

    def split(
        self,
        text: str,
    ) -> list[str]:
        """
        Splits a document into sentences.

        Parameters
        ----------
        text
            Clean document.

        Returns
        -------
        List of cleaned sentences.
        """

        if not text.strip():
            return []

        doc = self._nlp(text)

        sentences: list[str] = []

        for sentence in doc.sents:

            value = sentence.text.strip()

            if not value:
                continue

            sentences.append(value)

        return sentences

    def iter_sentences(
        self,
        text: str,
    ) -> Iterable[str]:
        """
        Lazily yields sentences.

        Useful for very large datasets.
        """

        if not text.strip():
            return

        doc = self._nlp(text)

        for sentence in doc.sents:

            value = sentence.text.strip()

            if value:
                yield value