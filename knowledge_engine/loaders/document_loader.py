from __future__ import annotations

from abc import ABC, abstractmethod

from knowledge_engine.domain.document.document_source import DocumentSource
from knowledge_engine.domain.document.raw_document import RawDocument


class DocumentLoader(ABC):
    """
    Base class for all document loaders.
    """

    @abstractmethod
    def load(self, source: DocumentSource) -> RawDocument:
        """
        Reads a document from its source.

        Parameters
        ----------
        source:
            Origin of the document.

        Returns
        -------
        RawDocument
            The raw content read from the source.
        """
        raise NotImplementedError