from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from types import MappingProxyType
from typing import Any, Mapping

from knowledge_engine.utils.id_generator import generate_id


class EntityType(Enum):
    UNKNOWN = auto()

    PERSON = auto()
    ORGANIZATION = auto()
    LOCATION = auto()

    PRODUCT = auto()
    DEVICE = auto()
    COMPONENT = auto()

    CONCEPT = auto()
    PROCEDURE = auto()
    MATERIAL = auto()

    SOFTWARE = auto()
    HARDWARE = auto()

    EVENT = auto()

    OTHER = auto()


@dataclass(frozen=True, slots=True)
class Entity:
    """
    Represents a semantic entity extracted from one or more documents.
    """

    name: str

    type: EntityType = EntityType.UNKNOWN

    description: str = ""

    metadata: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )

    id: str = field(default_factory=generate_id)