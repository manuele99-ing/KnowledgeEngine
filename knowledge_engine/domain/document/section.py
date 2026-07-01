from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from knowledge_engine.domain.document.block import Block
from knowledge_engine.utils.id_generator import generate_id


class SectionType(Enum):
    """
    Represents the structural type of a document section.
    """

    DOCUMENT = auto()
    CHAPTER = auto()
    SECTION = auto()
    SUBSECTION = auto()
    APPENDIX = auto()


@dataclass(frozen=True, slots=True)
class Section:
    """
    Logical section of a document.
    """

    title: str
    type: SectionType

    blocks: tuple[Block, ...] = ()

    children: tuple["Section", ...] = ()

    id: str = field(default_factory=generate_id)