from __future__ import annotations

from knowledge_engine.parsers.context.parse_context import ParseContext
from knowledge_engine.pipeline.pipeline_component import PipelineComponent


class ExtractTitleComponent(PipelineComponent):
    """
    Extracts the document title from the first non-empty line.
    """

    def process(self, data: ParseContext) -> ParseContext:
        lines = data.normalized_text.splitlines()

        title_index = next(
            (
                index
                for index, line in enumerate(lines)
                if line.strip()
            ),
            None,
        )

        if title_index is None:
            data.title = "Untitled"
            data.body_text = ""
            return data

        data.title = lines[title_index].strip()
        data.body_text = "\n".join(lines[title_index + 1 :])

        return data
