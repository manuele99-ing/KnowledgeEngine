from __future__ import annotations

from collections.abc import Iterator
from typing import Iterable

from knowledge_engine.language.datasets.provider import DatasetProvider
from knowledge_engine.language.preprocessing.cleaner import Cleaner
from knowledge_engine.language.preprocessing.duplicate_remover import DuplicateRemover
from knowledge_engine.language.preprocessing.language_filter import LanguageFilter
from knowledge_engine.language.preprocessing.sentence_splitter import SentenceSplitter


class CorpusBuilder:
    """
    Builds the language corpus.

    The builder orchestrates the complete preprocessing pipeline.

    Pipeline

        Provider
            ↓
        Cleaner
            ↓
        Language Filter
            ↓
        Sentence Splitter
            ↓
        Duplicate Remover
            ↓
        Corpus Stream

    The corpus is never fully loaded into memory.
    Sentences are produced lazily.
    """

    def __init__(
        self,
        cleaner: Cleaner,
        splitter: SentenceSplitter,
        duplicate_remover: DuplicateRemover,
        language_filter: LanguageFilter,
    ) -> None:

        self._cleaner = cleaner
        self._splitter = splitter
        self._duplicate_remover = duplicate_remover
        self._language_filter = language_filter

    def build(
        self,
        providers: Iterable[DatasetProvider],
    ) -> Iterator[str]:
        """
        Builds a unified language corpus.

        Returns
        -------
        Iterator[str]

            Lazy stream of corpus sentences.
        """

        return self._duplicate_remover.remove(
            self._sentence_stream(
                providers
            )
        )

    def _sentence_stream(
        self,
        providers: Iterable[DatasetProvider],
    ) -> Iterator[str]:

        for provider in providers:

            provider.download()

            provider.parse()

            for document in provider.documents():

                cleaned = self._cleaner.clean(
                    document
                )

                if not cleaned:
                    continue

                if not self._language_filter.accepts(
                    cleaned
                ):
                    continue

                yield from self._splitter.iter_sentences(
                    cleaned
                )