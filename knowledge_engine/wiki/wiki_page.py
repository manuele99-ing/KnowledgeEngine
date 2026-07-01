from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class WikiPage:
    """
    Represents a single page in the generated wiki.

    A page is an immutable representation of a knowledge item
    (entity, document, procedure, etc.) independently of how it
    will later be exported (Markdown, HTML, Obsidian, ...).
    """

    title: str

    content: str

    metadata: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )