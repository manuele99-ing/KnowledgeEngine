from __future__ import annotations

from knowledge_engine.parsers.context.parse_context import ParseContext
from knowledge_engine.pipeline.pipeline_component import PipelineComponent


class NormalizeTextComponent(PipelineComponent):
    """
    Performs basic text normalization.
    """

    def process(self, data: ParseContext) -> ParseContext:
        lines = [line.rstrip() for line in data.raw_document.content.splitlines()]

        data.normalized_text = "\n".join(lines).strip()

        return data
