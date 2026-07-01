from __future__ import annotations

from knowledge_engine.parsers.context.parse_context import ParseContext
from knowledge_engine.pipeline.pipeline_component import PipelineComponent


class SplitParagraphsComponent(PipelineComponent):
    """
    Splits body text into paragraph-like units.
    """

    def process(self, data: ParseContext) -> ParseContext:
        paragraphs: list[str] = []
        current: list[str] = []

        for line in data.body_text.splitlines():
            if line.strip():
                current.append(line.strip())
                continue

            if current:
                paragraphs.append("\n".join(current))
                current = []

        if current:
            paragraphs.append("\n".join(current))

        data.paragraphs = paragraphs

        return data
