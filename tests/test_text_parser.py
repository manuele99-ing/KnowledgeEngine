from __future__ import annotations

import unittest

from knowledge_engine.domain.document.block import BlockType
from knowledge_engine.domain.document.document_source import (
    DocumentSource,
    SourceType,
)
from knowledge_engine.domain.document.raw_document import RawDocument
from knowledge_engine.parsers.text_parser import TextParser


class TextParserTest(unittest.TestCase):
    def test_parse_builds_document_from_plain_text(self) -> None:
        source = DocumentSource(
            type=SourceType.TEXT,
            name="manuale.txt",
            uri="test_data/manuale.txt",
        )
        raw_document = RawDocument(
            source=source,
            content="Manuale Hub 2\n\nPrima sezione.\nAncora testo.\n\nSeconda sezione.",
        )

        document = TextParser().parse(raw_document)

        self.assertEqual(document.source, source)
        self.assertEqual(document.title, "Manuale Hub 2")
        self.assertEqual(len(document.sections), 1)

        section = document.sections[0]

        self.assertEqual(section.title, "Content")
        self.assertEqual(len(section.blocks), 2)
        self.assertEqual(section.blocks[0].type, BlockType.TEXT)
        self.assertEqual(section.blocks[0].content, "Prima sezione.\nAncora testo.")
        self.assertEqual(section.blocks[1].content, "Seconda sezione.")

    def test_parse_empty_document_uses_untitled(self) -> None:
        source = DocumentSource(
            type=SourceType.TEXT,
            name="empty.txt",
            uri="test_data/empty.txt",
        )
        raw_document = RawDocument(source=source, content="  \n\n")

        document = TextParser().parse(raw_document)

        self.assertEqual(document.title, "Untitled")
        self.assertEqual(len(document.sections), 1)
        self.assertEqual(document.sections[0].blocks, ())

    def test_parse_builds_sections_from_headings(self) -> None:
        source = DocumentSource(
            type=SourceType.TEXT,
            name="manuale.txt",
            uri="test_data/manuale.txt",
        )
        raw_document = RawDocument(
            source=source,
            content=(
                "Manuale Hub 2\n\n"
                "Installazione\n\n"
                "Collegare il dispositivo alla corrente.\n\n"
                "# Configurazione\n\n"
                "Aprire l'app Ajax."
            ),
        )

        document = TextParser().parse(raw_document)

        self.assertEqual(document.title, "Manuale Hub 2")
        self.assertEqual(len(document.sections), 2)
        self.assertEqual(document.sections[0].title, "Installazione")
        self.assertEqual(document.sections[1].title, "Configurazione")
        self.assertEqual(document.sections[0].blocks[0].type, BlockType.TEXT)
        self.assertEqual(
            document.sections[0].blocks[0].content,
            "Collegare il dispositivo alla corrente.",
        )
        self.assertEqual(
            document.sections[1].blocks[0].content,
            "Aprire l'app Ajax.",
        )

    def test_parse_builds_nested_sections_from_numbered_headings(self) -> None:
        source = DocumentSource(
            type=SourceType.TEXT,
            name="manuale.txt",
            uri="test_data/manuale.txt",
        )
        raw_document = RawDocument(
            source=source,
            content=(
                "Manuale Hub 2\n\n"
                "1. Installazione\n\n"
                "1.1 Requisiti\n\n"
                "- Hub 2 acceso\n"
                "- Connessione Internet\n\n"
                "1.2 Procedura\n\n"
                "1.2.1 Verifica\n\n"
                "Collegare il dispositivo alla corrente.\n\n"
                "2. Configurazione\n\n"
                "Aprire l'app Ajax."
            ),
        )

        document = TextParser().parse(raw_document)

        self.assertEqual(len(document.sections), 2)

        installation = document.sections[0]
        self.assertEqual(installation.title, "Installazione")
        self.assertEqual(len(installation.children), 2)
        self.assertEqual(installation.children[0].title, "Requisiti")
        self.assertEqual(installation.children[1].title, "Procedura")
        self.assertEqual(len(installation.children[1].children), 1)
        self.assertEqual(installation.children[1].children[0].title, "Verifica")
        self.assertEqual(
            installation.children[0].blocks[0].type,
            BlockType.LIST,
        )
        self.assertEqual(
            installation.children[1].children[0].blocks[0].content,
            "Collegare il dispositivo alla corrente.",
        )

    def test_parse_detects_list_blocks(self) -> None:
        source = DocumentSource(
            type=SourceType.TEXT,
            name="manuale.txt",
            uri="test_data/manuale.txt",
        )
        raw_document = RawDocument(
            source=source,
            content=(
                "Manuale Hub 2\n\n"
                "Requisiti\n\n"
                "- Hub 2 acceso\n"
                "- Connessione Internet\n"
                "- App Ajax installata\n\n"
                "Procedura\n\n"
                "1. Aprire l'app\n"
                "2. Selezionare il dispositivo"
            ),
        )

        document = TextParser().parse(raw_document)

        self.assertEqual(len(document.sections), 2)
        self.assertEqual(document.sections[0].title, "Requisiti")
        self.assertEqual(document.sections[0].blocks[0].type, BlockType.LIST)
        self.assertEqual(
            document.sections[0].blocks[0].content,
            "Hub 2 acceso\nConnessione Internet\nApp Ajax installata",
        )
        self.assertEqual(document.sections[1].title, "Procedura")
        self.assertEqual(document.sections[1].blocks[0].type, BlockType.LIST)
        self.assertEqual(
            document.sections[1].blocks[0].content,
            "Aprire l'app\nSelezionare il dispositivo",
        )


if __name__ == "__main__":
    unittest.main()
