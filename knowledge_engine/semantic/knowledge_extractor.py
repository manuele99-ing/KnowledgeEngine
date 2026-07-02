from __future__ import annotations

import re
from dataclasses import replace
from typing import Iterable

import spacy
from spacy.language import Language

from knowledge_engine.domain.document.block import Block
from knowledge_engine.domain.document.document import Document
from knowledge_engine.domain.document.section import Section
from knowledge_engine.domain.knowledge.entity import Entity, EntityType
from knowledge_engine.domain.knowledge.knowledge_base import KnowledgeBase
from knowledge_engine.domain.knowledge.relation import Relation, RelationType


class KnowledgeExtractor:
    """
    Extracts a first structured knowledge base from parsed documents.

    This extractor is intentionally heuristic and fully local. It creates the
    initial bridge between structured documents and the knowledge domain.
    """

    _candidate_pattern = re.compile(
        r"\b[A-Z][A-Za-z0-9]*(?:\s+[A-Z0-9][A-Za-z0-9]*)*\b"
    )
    _leading_stop_words = {
        "A",
        "An",
        "And",
        "Il",
        "La",
        "Le",
        "Lo",
        "The",
        "Un",
        "Una",
    }
    _ignored_candidates = {
        "Content",
        "Untitled",
    }
    _procedure_words = {
        "configurazione",
        "installazione",
        "procedura",
        "procedure",
        "setup",
    }
    _entity_label_type_map = {
        "PER": EntityType.PERSON,
        "LOC": EntityType.LOCATION,
        "GPE": EntityType.LOCATION,
        "ORG": EntityType.ORGANIZATION,
        "PRODUCT": EntityType.PRODUCT,
        "EVENT": EntityType.EVENT,
    }
    _relation_patterns = (
        (
            re.compile(
                r"\b([A-Z][A-Za-z0-9]*(?:\s+[A-Z0-9][A-Za-z0-9]*)*)\s+"
                r"(uses|usa|utilizza|utilizzano)\s+"
                r"([A-Z][A-Za-z0-9]*(?:\s+[A-Z0-9][A-Za-z0-9]*)*)",
                re.I,
            ),
            RelationType.USES,
        ),
        (
            re.compile(
                r"\b([A-Z][A-Za-z0-9]*(?:\s+[A-Z0-9][A-Za-z0-9]*)*)\s+"
                r"(connects to|si collega a|collega a)\s+"
                r"([A-Z][A-Za-z0-9]*(?:\s+[A-Z0-9][A-Za-z0-9]*)*)",
                re.I,
            ),
            RelationType.CONNECTED_TO,
        ),
        (
            re.compile(
                r"\b([A-Z][A-Za-z0-9]*(?:\s+[A-Z0-9][A-Za-z0-9]*)*)\s+"
                r"(is|è)\s+"
                r"([A-Z][A-Za-z0-9]*(?:\s+[A-Z0-9][A-Za-z0-9]*)*)",
                re.I,
            ),
            RelationType.IS_A,
        ),
        (
            re.compile(
                r"\b([A-Z][A-Za-z0-9]*(?:\s+[A-Z0-9][A-Za-z0-9]*)*)\s+"
                r"(part of|fa parte di|parte di)\s+"
                r"([A-Z][A-Za-z0-9]*(?:\s+[A-Z0-9][A-Za-z0-9]*)*)",
                re.I,
            ),
            RelationType.PART_OF,
        ),
        (
            re.compile(
                r"\b([A-Z][A-Za-z0-9]*(?:\s+[A-Z0-9][A-Za-z0-9]*)*)\s+"
                r"(located in|situato in|si trova in)\s+"
                r"([A-Z][A-Za-z0-9]*(?:\s+[A-Z0-9][A-Za-z0-9]*)*)",
                re.I,
            ),
            RelationType.LOCATED_IN,
        ),
    )

    def __init__(self) -> None:
        try:
            self._nlp = spacy.load("it_core_news_md")
        except OSError:
            self._nlp = spacy.load("it_core_news_sm")

    def extract(self, documents: Document | Iterable[Document]) -> KnowledgeBase:
        document_list = self._normalize_documents(documents)
        entities: dict[str, Entity] = {}
        relations: dict[tuple[str, RelationType, str], Relation] = {}

        for document in document_list:
            document_entity = self._add_entity(
                entities=entities,
                name=document.title,
                type=EntityType.CONCEPT,
            )

            for section in document.sections:
                self._extract_section(
                    section=section,
                    parent_entity=document_entity,
                    entities=entities,
                    relations=relations,
                )

        return KnowledgeBase(
            documents=tuple(document_list),
            entities=tuple(entities.values()),
            relations=tuple(relations.values()),
        )

    def _normalize_documents(
        self,
        documents: Document | Iterable[Document],
    ) -> list[Document]:
        if isinstance(documents, Document):
            return [documents]

        return list(documents)

    def _extract_section(
        self,
        section: Section,
        parent_entity: Entity,
        entities: dict[str, Entity],
        relations: dict[tuple[str, RelationType, str], Relation],
    ) -> None:
        if section.title in self._ignored_candidates:
            section_entity = parent_entity
        else:
            section_entity = self._add_entity(
                entities=entities,
                name=section.title,
                type=self._section_entity_type(section.title),
            )

            self._add_relation(
                relations=relations,
                source=parent_entity,
                target=section_entity,
                type=RelationType.CONTAINS,
            )

        for block in section.blocks:
            self._extract_block_entities(
                block=block,
                section_entity=section_entity,
                entities=entities,
                relations=relations,
            )

        for child in section.children:
            self._extract_section(
                section=child,
                parent_entity=section_entity,
                entities=entities,
                relations=relations,
            )

    def _extract_block_entities(
        self,
        block: Block,
        section_entity: Entity,
        entities: dict[str, Entity],
        relations: dict[tuple[str, RelationType, str], Relation],
    ) -> None:
        content = block.content.strip()

        if not content:
            return

        self._extract_spacy_entities(
            text=content,
            section_entity=section_entity,
            entities=entities,
            relations=relations,
        )

        for candidate in self._extract_candidates(content):
            entity = self._add_entity(
                entities=entities,
                name=candidate,
                type=EntityType.UNKNOWN,
            )
            self._add_relation(
                relations=relations,
                source=section_entity,
                target=entity,
                type=RelationType.REFERENCES,
            )

        self._extract_relations_from_text(
            text=content,
            entities=entities,
            relations=relations,
        )

    def _extract_spacy_entities(
        self,
        text: str,
        section_entity: Entity,
        entities: dict[str, Entity],
        relations: dict[tuple[str, RelationType, str], Relation],
    ) -> None:
        for entity in self._nlp(text).ents:
            candidate = self._clean_candidate(entity.text)

            if not candidate or candidate in self._ignored_candidates:
                continue

            entity_type = self._entity_type_from_label(entity.label_)
            extracted_entity = self._add_entity(
                entities=entities,
                name=candidate,
                type=entity_type,
            )
            self._add_relation(
                relations=relations,
                source=section_entity,
                target=extracted_entity,
                type=RelationType.REFERENCES,
            )

    def _extract_relations_from_text(
        self,
        text: str,
        entities: dict[str, Entity],
        relations: dict[tuple[str, RelationType, str], Relation],
    ) -> None:
        for pattern, relation_type in self._relation_patterns:
            for match in pattern.finditer(text):
                source_name = self._clean_candidate(match.group(1))
                target_name = self._clean_candidate(match.group(3))

                if not source_name or not target_name:
                    continue

                source_entity = self._add_entity(
                    entities=entities,
                    name=source_name,
                    type=EntityType.UNKNOWN,
                )
                target_entity = self._add_entity(
                    entities=entities,
                    name=target_name,
                    type=EntityType.UNKNOWN,
                )
                self._add_relation(
                    relations=relations,
                    source=source_entity,
                    target=target_entity,
                    type=relation_type,
                )

    def _extract_candidates(self, text: str) -> list[str]:
        candidates: list[str] = []

        for match in self._candidate_pattern.finditer(text):
            candidate = self._clean_candidate(match.group(0))

            if not candidate or candidate in self._ignored_candidates:
                continue

            if candidate not in candidates:
                candidates.append(candidate)

        return candidates

    def _clean_candidate(self, candidate: str) -> str:
        tokens = candidate.strip().split()

        while tokens and tokens[0] in self._leading_stop_words:
            tokens.pop(0)

        return " ".join(tokens).strip()

    def _add_entity(
        self,
        entities: dict[str, Entity],
        name: str,
        type: EntityType,
    ) -> Entity:
        candidate = name.strip()

        if not candidate or candidate in self._ignored_candidates:
            candidate = "Unknown"

        key = candidate.casefold()
        existing = entities.get(key)

        if existing is None:
            entity = Entity(name=candidate, type=type)
            entities[key] = entity
            return entity

        if existing.type is EntityType.UNKNOWN and type is not EntityType.UNKNOWN:
            entity = replace(existing, type=type)
            entities[key] = entity
            return entity

        return existing

    def _add_relation(
        self,
        relations: dict[tuple[str, RelationType, str], Relation],
        source: Entity,
        target: Entity,
        type: RelationType,
    ) -> None:
        if source.name == target.name:
            return

        key = (source.name.casefold(), type, target.name.casefold())

        if key in relations:
            return

        relations[key] = Relation(
            source=source,
            target=target,
            type=type,
        )

    def _entity_type_from_label(self, label: str) -> EntityType:
        return self._entity_label_type_map.get(label, EntityType.UNKNOWN)

    def _section_entity_type(self, title: str) -> EntityType:
        title_words = {word.casefold() for word in title.split()}

        if title_words & self._procedure_words:
            return EntityType.PROCEDURE

        return EntityType.CONCEPT
