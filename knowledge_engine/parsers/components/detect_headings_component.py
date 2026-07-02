from __future__ import annotations

import re

from knowledge_engine.parsers.context.parse_context import ParseContext
from knowledge_engine.pipeline.pipeline_component import PipelineComponent


class DetectHeadingsComponent(PipelineComponent):
    """
    Detects heading-like paragraphs in plain text using multiple signals.
    """

    _sentence_endings = (".", "?", "!", ";")
    _markdown_heading_pattern = re.compile(r"^(#{1,6})\s*(.+)$")
    _numeric_heading_pattern = re.compile(r"^(\d+(?:\.\d+)*)(?:[\.\)])?\s+(.+)$")
    _list_item_pattern = re.compile(r"^(?:[-*+]\s+|\d+(?:\.\d+)*[\.)]\s+).+$")
    _heading_score_threshold = 4

    def process(self, data: ParseContext) -> ParseContext:
        from knowledge_engine.parsers.context.parse_context import Paragraph

        updated_paragraphs: list[Paragraph] = []
        heading_indexes: set[int] = set()
        heading_levels: dict[int, int] = {}

        for index, paragraph in enumerate(data.paragraphs):
            heading_level, heading_style = self._heading_info(
                paragraph,
                previous_paragraph=data.paragraphs[index - 1] if index > 0 else None,
                next_paragraph=data.paragraphs[index + 1] if index + 1 < len(data.paragraphs) else None,
                position=index,
                total=len(data.paragraphs),
            )

            is_heading = heading_level is not None

            if is_heading:
                heading_indexes.add(index)
                heading_levels[index] = heading_level

            updated_paragraphs.append(
                type(paragraph)(
                    text=paragraph.text,
                    original_lines=paragraph.original_lines,
                    start_line=paragraph.start_line,
                    end_line=paragraph.end_line,
                    indentation=paragraph.indentation,
                    blank_lines_before=paragraph.blank_lines_before,
                    blank_lines_after=paragraph.blank_lines_after,
                    heading_level=heading_level,
                    heading_style=heading_style,
                    is_heading=is_heading,
                )
            )

        data.paragraphs = updated_paragraphs
        data.heading_indexes = heading_indexes
        data.heading_levels = heading_levels

        return data

    def _heading_info(
        self,
        paragraph,
        previous_paragraph,
        next_paragraph,
        position: int,
        total: int,
    ) -> tuple[int | None, str | None]:
        text = paragraph.text.strip()

        if not text or "\n" in text:
            return None, None

        if self._is_list_paragraph(paragraph, previous_paragraph, next_paragraph):
            return None, None

        markdown_match = self._markdown_heading_pattern.match(text)
        if markdown_match:
            return len(markdown_match.group(1)), "markdown"

        numeric_match = self._numeric_heading_pattern.match(text)
        if numeric_match and (paragraph.blank_lines_after > 0 or next_paragraph is not None):
            return len(numeric_match.group(1).split(".")), "numeric"

        return self._implicit_heading_info(
            paragraph,
            previous_paragraph,
            next_paragraph,
            position,
            total,
        )

    def _is_list_paragraph(self, paragraph, previous_paragraph, next_paragraph) -> bool:
        lines = [line.strip() for line in paragraph.original_lines if line.strip()]
        if not lines:
            return False

        matches = [bool(self._list_item_pattern.match(line)) for line in lines]
        if not all(matches):
            return False

        if len(lines) > 1:
            return True

        if previous_paragraph is not None:
            previous_lines = [line.strip() for line in previous_paragraph.original_lines if line.strip()]
            if any(self._list_item_pattern.match(line) for line in previous_lines):
                return True

        if next_paragraph is not None:
            next_lines = [line.strip() for line in next_paragraph.original_lines if line.strip()]
            if any(self._list_item_pattern.match(line) for line in next_lines):
                return True

        single_line = lines[0]
        if single_line.startswith(("- ", "* ", "+ ")):
            return True

        return False

    def _implicit_heading_info(
        self,
        paragraph,
        previous_paragraph,
        next_paragraph,
        position: int,
        total: int,
    ) -> tuple[int | None, str | None]:
        text = paragraph.text.strip()

        if not text:
            return None, None

        if len(text) > 80:
            return None, None

        if text[-1] in self._sentence_endings:
            return None, None

        if not text[0].isupper():
            return None, None

        score = 0

        if paragraph.blank_lines_before > 0:
            score += 2

        if paragraph.blank_lines_after > 0:
            score += 2

        if text.endswith(":"):
            score += 2

        if text.isupper() and len(text.split()) <= 5:
            score += 2

        if len(text.split()) <= 4:
            score += 1

        if next_paragraph is not None:
            next_text = next_paragraph.text.strip()
            if len(next_text) > 40:
                score += 1
            if next_text[0].isupper():
                score += 1
            if next_text.endswith(self._sentence_endings):
                score += 1
            if any(line.strip().startswith(('-', '*', '+')) for line in next_paragraph.original_lines):
                score += 1

        if previous_paragraph is not None and previous_paragraph.is_heading:
            score += 1

        return (1, "implicit") if score >= self._heading_score_threshold else (None, None)
