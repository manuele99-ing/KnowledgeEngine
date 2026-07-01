from __future__ import annotations

from dataclasses import dataclass

from knowledge_engine.retrieval.indexing.indexed_section import IndexedSection


@dataclass(frozen=True, slots=True)
class RetrievalMatch:
    """
    Represents a ranked retrieval result produced by a retriever.
    """

    section: IndexedSection

    score: float