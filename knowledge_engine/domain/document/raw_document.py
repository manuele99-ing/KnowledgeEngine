from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from knowledge_engine.domain.document.document_source import DocumentSource
from knowledge_engine.utils.id_generator import generate_id


@dataclass(frozen=True, slots=True)
class RawDocument:
    """
    Represents a document immediately after loading.

    The content has been read from the source but has not yet been
    parsed into a structured Document.
    """

    source: DocumentSource

    content: str

    metadata: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )

    id: str = field(default_factory=generate_id)