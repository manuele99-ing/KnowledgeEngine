from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from types import MappingProxyType
from typing import Any, Mapping

from knowledge_engine.domain.knowledge.entity import Entity
from knowledge_engine.utils.id_generator import generate_id


class RelationType(Enum):
    UNKNOWN = auto()

    IS_A = auto()
    PART_OF = auto()
    HAS_PART = auto()

    CONNECTED_TO = auto()
    DEPENDS_ON = auto()

    LOCATED_IN = auto()
    BELONGS_TO = auto()

    USES = auto()
    CREATES = auto()
    CONTAINS = auto()

    REFERENCES = auto()


@dataclass(frozen=True, slots=True)
class Relation:
    """
    Represents a semantic relationship between two entities.
    """

    source: Entity

    target: Entity

    type: RelationType

    description: str = ""

    metadata: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({}))

    id: str = field(default_factory=generate_id)