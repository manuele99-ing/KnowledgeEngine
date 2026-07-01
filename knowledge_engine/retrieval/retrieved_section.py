from __future__ import annotations

from dataclasses import dataclass

from knowledge_engine.domain.document.document import Document
from knowledge_engine.domain.document.section import Section


@dataclass(frozen=True, slots=True)
class RetrievedSection:
    """
    A section selected by the retrieval engine.
    """

    document: Document

    section: Section

    score: float