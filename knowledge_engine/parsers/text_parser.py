from __future__ import annotations

from knowledge_engine.domain.document.block import Block, BlockType
from knowledge_engine.domain.document.document import Document
from knowledge_engine.domain.document.raw_document import RawDocument
from knowledge_engine.domain.document.section import Section, SectionType
from knowledge_engine.parsers.components.normalize_text_component import (
    NormalizeTextComponent,
)
from knowledge_engine.parsers.document_parser import DocumentParser
from knowledge_engine.pipeline.pipeline import Pipeline


class TextParser(DocumentParser):
    """
    Plain text parser.
    """

    def __init__(self) -> None:

        self._pipeline = Pipeline()

        self._pipeline.add_component(
            NormalizeTextComponent()
        )

    def parse(self, raw_document: RawDocument) -> Document:

        raw_document = self._pipeline.run(raw_document)

        lines = [
            line
            for line in raw_document.content.splitlines()
            if line.strip()
        ]

        title = lines[0] if lines else "Untitled"

        body = "\n".join(lines[1:])

        block = Block(
            type=BlockType.TEXT,
            content=body,
        )

        section = Section(
            title="Content",
            type=SectionType.CHAPTER,
            blocks=(block,),
        )

        return Document(
            source=raw_document.source,
            title=title,
            sections=(section,),
        )