from __future__ import annotations

import re

from knowledge_engine.parsers.context.parse_context import ParseContext
from knowledge_engine.pipeline.pipeline_component import PipelineComponent


class DetectListsComponent(PipelineComponent):
    """
    Detects simple unordered and ordered list paragraphs.
    """

    _ordered_item_pattern = re.compile(r"^\d+[\.)]\s+")
    _unordered_markers = ("- ", "* ", "+ ")

    def process(self, data: ParseContext) -> ParseContext:
        data.list_indexes = {
            index
            for index, paragraph in enumerate(data.paragraphs)
            if index not in data.heading_indexes and self._is_list(paragraph)
        }

        return data

    def _is_list(self, paragraph: str) -> bool:
        lines = [
            line.strip()
            for line in paragraph.splitlines()
            if line.strip()
        ]

        if not lines:
            return False

        return all(self._is_list_item(line) for line in lines)

    def _is_list_item(self, line: str) -> bool:
        return line.startswith(self._unordered_markers) or bool(
            self._ordered_item_pattern.match(line)
        )
