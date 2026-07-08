from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path


class DatasetProvider(ABC):
    """
    Base interface implemented by every language dataset provider.

    A provider is responsible for acquiring a public dataset and exposing
    its textual content to the Language module.

    Every provider follows the same lifecycle:

        download()
            Downloads the raw dataset if it is not already available.

        parse()
            Reads the raw dataset and prepares it for iteration.

        documents()
            Lazily yields the textual documents contained in the dataset.

    Providers never perform:

        - text cleaning
        - sentence splitting
        - duplicate removal
        - language detection
        - model training

    Those responsibilities belong to the preprocessing and
    training pipelines.
    """

    def __init__(
        self,
        dataset_root: Path,
    ) -> None:

        self._dataset_root = dataset_root

    @property
    def dataset_root(
        self,
    ) -> Path:
        """
        Local directory where the dataset is stored.
        """
        return self._dataset_root

    @property
    @abstractmethod
    def name(
        self,
    ) -> str:
        """
        Human readable dataset name.
        """

    @abstractmethod
    def download(
        self,
        force: bool = False,
    ) -> None:
        """
        Downloads the dataset.

        Parameters
        ----------
        force
            Forces a new download even if the dataset already exists.
        """

    @abstractmethod
    def parse(
        self,
    ) -> None:
        """
        Parses the downloaded dataset and prepares it for
        document iteration.
        """

    @abstractmethod
    def documents(
        self,
    ) -> Iterator[str]:
        """
        Lazily yields textual documents.

        Each yielded string represents one logical document
        (article, page, manual section, etc.).

        The provider must not load the entire dataset into memory
        if this can be avoided.
        """

    def exists(
        self,
    ) -> bool:
        """
        Returns True if the dataset already exists locally.
        """

        return self.dataset_root.exists()