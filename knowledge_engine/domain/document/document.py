from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from knowledge_engine.domain.document.document_source import DocumentSource
from knowledge_engine.domain.document.section import Section
from knowledge_engine.utils.id_generator import generate_id


@dataclass(frozen=True, slots=True)
class Document:
    """
    Represents a structured document.

    A Document is composed of one or more sections and originates
    from a specific DocumentSource.
    """

    source: DocumentSource

    title: str

    sections: tuple[Section, ...] = ()

    metadata: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )

    id: str = field(default_factory=generate_id)