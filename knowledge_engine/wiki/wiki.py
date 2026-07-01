from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from knowledge_engine.wiki.wiki_index import WikiIndex
from knowledge_engine.wiki.wiki_page import WikiPage


@dataclass(frozen=True, slots=True)
class Wiki:
    """
    Represents an immutable generated wiki.

    A Wiki is the logical representation of the documentation
    generated from a KnowledgeBase.

    It is independent from any export format
    (Markdown, HTML, MkDocs, Obsidian, ...).
    """

    index: WikiIndex

    pages: tuple[WikiPage, ...] = field(default_factory=tuple)

    metadata: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )