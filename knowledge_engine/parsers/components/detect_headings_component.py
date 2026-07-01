from __future__ import annotations

from knowledge_engine.parsers.context.parse_context import ParseContext
from knowledge_engine.pipeline.pipeline_component import PipelineComponent


class DetectHeadingsComponent(PipelineComponent):
    """
    Detects simple heading-like paragraphs in plain text.
    """

    _sentence_endings = (".", ",", ";", "?", "!")

    def process(self, data: ParseContext) -> ParseContext:
        data.heading_indexes = {
            index
            for index, paragraph in enumerate(data.paragraphs)
            if self._is_heading(paragraph)
        }

        return data

    def _is_heading(self, paragraph: str) -> bool:
        text = paragraph.strip()

        if not text:
            return False

        if text.startswith("#"):
            return bool(text.lstrip("#").strip())

        if "\n" in text:
            return False

        if len(text) > 80:
            return False

        if not any(character.isalpha() for character in text):
            return False

        return not text.endswith(self._sentence_endings)
