from __future__ import annotations

from dataclasses import dataclass

from knowledge_engine.domain.document.document import Document
from knowledge_engine.domain.document.section import Section


@dataclass(frozen=True, slots=True)
class IndexedSection:
    """
    Section indexed by the retrieval engine.
    """

    document: Document

    section: Section

    text: str

    tokens: tuple[str, ...]

    path: tuple[str, ...] = ()

    vector: tuple[float, ...] = ()