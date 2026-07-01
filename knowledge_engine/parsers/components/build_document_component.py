from __future__ import annotations

from knowledge_engine.domain.document.document import Document
from knowledge_engine.parsers.context.parse_context import ParseContext
from knowledge_engine.pipeline.pipeline_component import PipelineComponent


class BuildDocumentComponent(PipelineComponent):
    """
    Builds the final immutable Document.
    """

    def process(self, data: ParseContext) -> ParseContext:
        data.document = Document(
            source=data.raw_document.source,
            title=data.title or "Untitled",
            sections=tuple(data.sections),
            metadata=data.raw_document.metadata,
        )

        return data
