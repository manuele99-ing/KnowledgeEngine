from __future__ import annotations

from pathlib import Path
from typing import Iterable

from knowledge_engine.language.datasets.provider import DatasetProvider


class HuggingFaceProvider(DatasetProvider):
    """
    Dataset provider based on Hugging Face datasets.

    The provider delegates downloading and parsing to the
    corresponding helper classes.

    It exposes a uniform interface to the corpus builder.
    """

    def __init__(
        self,
        dataset_root: Path,
        dataset_name: str,
        subset: str | None = None,
        split: str = "train",
    ) -> None:

        super().__init__(dataset_root)

        self._dataset_name = dataset_name
        self._subset = subset
        self._split = split

        self._documents: list[str] = []

    @property
    def name(
        self,
    ) -> str:

        return self._dataset_name

    def download(
        self,
        force: bool = False,
    ) -> None:

        #
        # implemented by downloader.py
        #

        ...

    def parse(
        self,
    ) -> None:

        #
        # implemented by parser.py
        #

        ...

    def documents(
        self,
    ) -> Iterable[str]:

        yield from self._documents