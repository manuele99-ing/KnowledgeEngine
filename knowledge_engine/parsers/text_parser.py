from __future__ import annotations

from knowledge_engine.domain.document.document import Document
from knowledge_engine.domain.document.raw_document import RawDocument
from knowledge_engine.parsers.components.build_blocks_component import (
    BuildBlocksComponent,
)
from knowledge_engine.parsers.components.build_document_component import (
    BuildDocumentComponent,
)
from knowledge_engine.parsers.components.build_sections_component import (
    BuildSectionsComponent,
)
from knowledge_engine.parsers.components.detect_headings_component import (
    DetectHeadingsComponent,
)
from knowledge_engine.parsers.components.detect_lists_component import (
    DetectListsComponent,
)
from knowledge_engine.parsers.components.extract_title_component import (
    ExtractTitleComponent,
)
from knowledge_engine.parsers.components.normalize_text_component import (
    NormalizeTextComponent,
)
from knowledge_engine.parsers.components.split_paragraphs_component import (
    SplitParagraphsComponent,
)
from knowledge_engine.parsers.components.validate_headings_component import (
    ValidateHeadingsComponent,
)
from knowledge_engine.parsers.context.parse_context import ParseContext
from knowledge_engine.parsers.document_parser import DocumentParser
from knowledge_engine.pipeline.pipeline import Pipeline


class TextParser(DocumentParser):
    """
    Plain text parser.
    """

    def __init__(self) -> None:

        self._pipeline = Pipeline()

        self._pipeline.add_component(NormalizeTextComponent())
        self._pipeline.add_component(ExtractTitleComponent())
        self._pipeline.add_component(SplitParagraphsComponent())
        self._pipeline.add_component(DetectHeadingsComponent())
        self._pipeline.add_component(ValidateHeadingsComponent())
        self._pipeline.add_component(DetectListsComponent())
        self._pipeline.add_component(BuildBlocksComponent())
        self._pipeline.add_component(BuildSectionsComponent())
        self._pipeline.add_component(BuildDocumentComponent())

    def parse(self, raw_document: RawDocument) -> Document:

        context = ParseContext(raw_document=raw_document)
        context = self._pipeline.run(context)

        if context.document is None:
            raise ValueError("Text parsing pipeline did not build a document.")

        return context.document
