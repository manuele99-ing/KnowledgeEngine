from __future__ import annotations

from pathlib import Path

from knowledge_engine.domain.document.document_source import DocumentSource
from knowledge_engine.domain.document.raw_document import RawDocument
from knowledge_engine.loaders.document_loader import DocumentLoader


class TextLoader(DocumentLoader):
    """
    Loads plain text documents.
    """

    def __init__(self, encoding: str = "utf-8") -> None:
        self._encoding = encoding

    def load(self, source: DocumentSource) -> RawDocument:
        path = Path(source.uri)

        content = path.read_text(encoding=self._encoding)

        return RawDocument(
            source=source,
            content=content,
        )