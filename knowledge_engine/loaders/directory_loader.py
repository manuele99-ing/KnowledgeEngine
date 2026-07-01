from __future__ import annotations

from pathlib import Path

from knowledge_engine.domain.document.document_source import (
    DocumentSource,
    SourceType,
)
from knowledge_engine.domain.document.raw_document import RawDocument
from knowledge_engine.loaders.document_loader import DocumentLoader
from knowledge_engine.loaders.text_loader import TextLoader


class DirectoryLoader:
    """
    Loads every supported document contained in a directory.
    """

    def __init__(self) -> None:
        self._text_loader = TextLoader()

    def load_directory(
        self,
        directory: str | Path,
    ) -> list[RawDocument]:

        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(
                f"Directory not found: {directory}"
            )

        if not directory.is_dir():
            raise NotADirectoryError(directory)

        documents: list[RawDocument] = []

        for path in sorted(directory.iterdir()):

            if not path.is_file():
                continue

            loader = self._select_loader(path)

            if loader is None:
                continue

            source = DocumentSource(
                type=self._source_type(path),
                name=path.name,
                uri=str(path),
            )

            documents.append(
                loader.load(source)
            )

        return documents

    def _select_loader(
        self,
        path: Path,
    ) -> DocumentLoader | None:

        match path.suffix.casefold():

            case ".txt":
                return self._text_loader

            case _:
                return None

    def _source_type(
        self,
        path: Path,
    ) -> SourceType:

        match path.suffix.casefold():

            case ".txt":
                return SourceType.TEXT

            case _:
                raise ValueError(
                    f"Unsupported file type: {path.suffix}"
                )