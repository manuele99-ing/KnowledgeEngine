from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from types import MappingProxyType
from typing import Any, Mapping

from knowledge_engine.utils.id_generator import generate_id


class SourceType(Enum):
    PDF = auto()
    DOCX = auto()
    HTML = auto()
    MARKDOWN = auto()
    TEXT = auto()
    EMAIL = auto()
    DATABASE = auto()
    OCR = auto()
    UNKNOWN = auto()


@dataclass(frozen=True, slots=True)
class DocumentSource:
    """
    Represents the origin of a document.

    It describes where a document comes from, independently from
    how it will be parsed or processed.
    """

    type: SourceType
    name: str
    uri: str

    metadata: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )

    id: str = field(default_factory=generate_id)