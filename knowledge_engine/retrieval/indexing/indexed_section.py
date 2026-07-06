from __future__ import annotations

from dataclasses import dataclass

from knowledge_engine.domain.document.document import Document
from knowledge_engine.domain.document.section import Section


@dataclass(frozen=True, slots=True)
class IndexedSection:
    """
    Section indexed by the retrieval engine.

    The retrieval pipeline keeps title, path and body separated so
    that the ranking algorithm can weight them independently.
    """

    document: Document

    section: Section

    title: str

    body: str

    text: str

    title_tokens: tuple[str, ...]

    body_tokens: tuple[str, ...]

    tokens: tuple[str, ...]

    path: tuple[str, ...]

    vector: tuple[float, ...] = ()