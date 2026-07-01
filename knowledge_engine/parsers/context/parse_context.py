from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from knowledge_engine.domain.document.block import Block
from knowledge_engine.domain.document.document import Document
from knowledge_engine.domain.document.raw_document import RawDocument
from knowledge_engine.domain.document.section import Section


@dataclass(slots=True)
class ParseContext:
    """
    Represents the temporary state of the parsing pipeline.

    The ParseContext is progressively enriched by parser components until
    a fully structured Document is produced.

    This object is NOT part of the domain model and exists only during
    the parsing process.
    """

    raw_document: RawDocument

    normalized_text: str = ""

    title: str | None = None

    body_text: str = ""

    paragraphs: list[str] = field(default_factory=list)

    heading_indexes: set[int] = field(default_factory=set)

    list_indexes: set[int] = field(default_factory=set)

    blocks: list[Block] = field(default_factory=list)

    sections: list[Section] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

    document: Document | None = None
