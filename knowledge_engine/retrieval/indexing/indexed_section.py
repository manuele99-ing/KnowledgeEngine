from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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

    #
    # Tokens estratti dal titolo.
    #

    title_tokens: tuple[str, ...]

    #
    # Token del solo body della sezione.
    #

    body_tokens: tuple[str, ...]

    #
    # Tutti i token utilizzati dal retrieval.
    #

    tokens: tuple[str, ...]

    #
    # Breadcrumb della sezione.
    #

    path: tuple[str, ...]

    #
    # Semantic embedding (BGE).
    #

    embedding: np.ndarray | None = None

    @property
    def path_text(
        self,
    ) -> str:

        return " > ".join(self.path)

    @property
    def all_text(
        self,
    ) -> str:

        return self.text

    @property
    def has_embedding(
        self,
    ) -> bool:

        return self.embedding is not None