from __future__ import annotations

from abc import ABC, abstractmethod

from knowledge_engine.domain.document.document import Document
from knowledge_engine.domain.document.raw_document import RawDocument


class DocumentParser(ABC):
    """
    Base class for all document parsers.
    """

    @abstractmethod
    def parse(self, raw_document: RawDocument) -> Document:
        raise NotImplementedError