from __future__ import annotations

import re

from knowledge_engine.parsers.context.parse_context import Paragraph, ParseContext
from knowledge_engine.pipeline.pipeline_component import PipelineComponent


class SplitParagraphsComponent(PipelineComponent):
    """
    Splits body text into paragraph-like units.
    """

    _heading_line_pattern = re.compile(r"^(#{1,6})\s+.+$|^\d+(?:\.\d+)*(?:[\.)])?\s+.+$|^[A-ZÈÉÀÙÌÒ]{2,}(?:\s+[A-ZÈÉÀÙÌÒ]+)*$|.+:$")
    _list_item_pattern = re.compile(r"^(?:[-*+]\s+|\d+(?:\.\d+)*[\.)]\s+).+$")

    def process(self, data: ParseContext) -> ParseContext:
        paragraphs: list[Paragraph] = []
        current_lines: list[str] = []
        current_start_line = 0
        blank_lines = 0
        blank_before = 0

        for line_number, line in enumerate(data.body_text.splitlines(), start=1):
            if line.strip():
                if not current_lines:
                    current_start_line = line_number
                    blank_before = blank_lines
                    current_indentation = len(line) - len(line.lstrip(" "))
                current_lines.append(line.rstrip())
                blank_lines = 0
                continue

            blank_lines += 1
            if current_lines:
                paragraphs.extend(
                    self._split_paragraph(
                        current_lines,
                        current_start_line,
                        blank_before,
                        blank_lines,
                        current_indentation,
                    )
                )
                current_lines = []

        if current_lines:
            paragraphs.extend(
                self._split_paragraph(
                    current_lines,
                    current_start_line,
                    blank_before,
                    0,
                    current_indentation,
                )
            )

        data.paragraphs = paragraphs

        return data

    def _split_paragraph(
        self,
        lines: list[str],
        start_line: int,
        blank_before: int,
        blank_after: int,
        indentation: int,
    ) -> list[Paragraph]:
        if len(lines) > 1 and self._is_heading_line(lines[0], lines[1]) and not self._is_list_paragraph(lines):
            heading_text = lines[0].strip()
            heading = Paragraph(
                text=heading_text,
                original_lines=(lines[0],),
                start_line=start_line,
                end_line=start_line,
                indentation=indentation,
                blank_lines_before=blank_before,
                blank_lines_after=0,
            )
            body_lines = [line.rstrip() for line in lines[1:]]
            body_indentation = len(body_lines[0]) - len(body_lines[0].lstrip(" "))
            body = Paragraph(
                text="\n".join(line.strip() for line in body_lines),
                original_lines=tuple(body_lines),
                start_line=start_line + 1,
                end_line=start_line + len(lines) - 1,
                indentation=body_indentation,
                blank_lines_before=0,
                blank_lines_after=blank_after,
            )
            return [heading, body]

        return [
            Paragraph(
                text="\n".join(line.strip() for line in lines),
                original_lines=tuple(lines),
                start_line=start_line,
                end_line=start_line + len(lines) - 1,
                indentation=indentation,
                blank_lines_before=blank_before,
                blank_lines_after=blank_after,
            )
        ]

    def _is_list_paragraph(self, lines: list[str]) -> bool:
        return all(self._list_item_pattern.match(line.strip()) for line in lines if line.strip())

    def _is_heading_line(self, first_line: str, second_line: str) -> bool:
        first = first_line.strip()
        if not first:
            return False

        if self._heading_line_pattern.match(first):
            return True

        if len(first.split()) <= 4 and first[0].isupper() and not first.endswith(
            "."
        ) and not first.endswith("?") and not first.endswith("!"):
            return True

        return False
