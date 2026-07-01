from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from knowledge_engine.utils.id_generator import generate_id


class BlockType(Enum):
    """
    Represents the semantic type of a document block.
    """

    TEXT = auto()
    HEADING = auto()
    LIST = auto()
    TABLE = auto()
    IMAGE = auto()
    CODE = auto()
    NOTE = auto()
    WARNING = auto()
    QUOTE = auto()


@dataclass(frozen=True, slots=True)
class Block:
    """
    Smallest semantic unit of a document.
    """

    type: BlockType
    content: str
    id: str = field(default_factory=generate_id)