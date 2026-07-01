from __future__ import annotations

from dataclasses import dataclass, field

from knowledge_engine.domain.document.document import Document
from knowledge_engine.domain.knowledge.entity import Entity
from knowledge_engine.domain.knowledge.relation import Relation
from knowledge_engine.utils.id_generator import generate_id


@dataclass(frozen=True, slots=True)
class KnowledgeBase:
    """
    Represents the structured knowledge extracted from one or more documents.
    """

    documents: tuple[Document, ...] = ()

    entities: tuple[Entity, ...] = ()

    relations: tuple[Relation, ...] = ()

    id: str = field(default_factory=generate_id)