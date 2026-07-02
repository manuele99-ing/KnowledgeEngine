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
            if index not in data.heading_indexes
            and self._is_list(
                paragraph,
                previous_paragraph=data.paragraphs[index - 1] if index > 0 else None,
                next_paragraph=data.paragraphs[index + 1] if index + 1 < len(data.paragraphs) else None,
            )
        }

        return data

    def _is_list(
        self,
        paragraph,
        previous_paragraph,
        next_paragraph,
    ) -> bool:
        lines = [
            line.strip()
            for line in paragraph.original_lines
            if line.strip()
        ]

        if not lines:
            return False

        if not all(self._is_list_item(line) for line in lines):
            return False

        if len(lines) > 1:
            return True

        line = lines[0]
        if line.startswith(self._unordered_markers):
            return True

        if paragraph.blank_lines_before == 0 and paragraph.blank_lines_after == 0:
            return True

        if next_paragraph is not None:
            next_lines = [
                line.strip()
                for line in next_paragraph.original_lines
                if line.strip()
            ]
            if any(self._is_list_item(next_line) for next_line in next_lines):
                return paragraph.blank_lines_before == 0 or paragraph.blank_lines_after == 0

        if previous_paragraph is not None:
            previous_lines = [
                line.strip()
                for line in previous_paragraph.original_lines
                if line.strip()
            ]
            if any(self._is_list_item(prev_line) for prev_line in previous_lines):
                return paragraph.blank_lines_before == 0 or paragraph.blank_lines_after == 0

        return False

    def _is_list_item(self, line: str) -> bool:
        return line.startswith(self._unordered_markers) or bool(
            self._ordered_item_pattern.match(line)
        )
