from __future__ import annotations

import unittest

from knowledge_engine.domain.document.document_source import (
    DocumentSource,
    SourceType,
)
from knowledge_engine.domain.document.raw_document import RawDocument
from knowledge_engine.domain.knowledge.entity import EntityType
from knowledge_engine.domain.knowledge.relation import RelationType
from knowledge_engine.parsers.text_parser import TextParser
from knowledge_engine.semantic.knowledge_extractor import KnowledgeExtractor


class KnowledgeExtractorTest(unittest.TestCase):
    def test_extract_builds_knowledge_base_from_document(self) -> None:
        document = self._parse(
            "Ajax Hub 2\n\n"
            "Installazione\n\n"
            "Hub 2 uses Jeweller.\n"
            "MotionProtect connects to Hub 2."
        )

        knowledge_base = KnowledgeExtractor().extract(document)

        entity_names = {entity.name for entity in knowledge_base.entities}
        relation_keys = {
            (relation.source.name, relation.type, relation.target.name)
            for relation in knowledge_base.relations
        }

        self.assertEqual(knowledge_base.documents, (document,))
        self.assertIn("Ajax Hub 2", entity_names)
        self.assertIn("Installazione", entity_names)
        self.assertIn("Hub 2", entity_names)
        self.assertIn("Jeweller", entity_names)
        self.assertIn("MotionProtect", entity_names)
        self.assertIn(
            ("Ajax Hub 2", RelationType.CONTAINS, "Installazione"),
            relation_keys,
        )
        self.assertIn(
            ("Installazione", RelationType.REFERENCES, "Jeweller"),
            relation_keys,
        )

        entity_types = {
            entity.name: entity.type
            for entity in knowledge_base.entities
        }

        self.assertEqual(entity_types["Installazione"], EntityType.PROCEDURE)

    def test_extract_does_not_create_content_fallback_entity(self) -> None:
        document = self._parse(
            "Ajax Hub 2\n\n"
            "The communication protocol is Jeweller."
        )

        knowledge_base = KnowledgeExtractor().extract(document)

        entity_names = {entity.name for entity in knowledge_base.entities}
        relation_keys = {
            (relation.source.name, relation.type, relation.target.name)
            for relation in knowledge_base.relations
        }

        self.assertNotIn("Content", entity_names)
        self.assertIn("Jeweller", entity_names)
        self.assertIn(
            ("Ajax Hub 2", RelationType.REFERENCES, "Jeweller"),
            relation_keys,
        )

    def _parse(self, content: str):
        source = DocumentSource(
            type=SourceType.TEXT,
            name="manuale.txt",
            uri="test_data/manuale.txt",
        )
        raw_document = RawDocument(source=source, content=content)

        return TextParser().parse(raw_document)


if __name__ == "__main__":
    unittest.main()
