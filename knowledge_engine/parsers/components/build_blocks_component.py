from __future__ import annotations

import re

from knowledge_engine.domain.document.block import Block, BlockType
from knowledge_engine.parsers.context.parse_context import ParseContext
from knowledge_engine.pipeline.pipeline_component import PipelineComponent


class BuildBlocksComponent(PipelineComponent):
    """
    Builds document blocks from parsed paragraphs.
    """

    _ordered_item_pattern = re.compile(r"^\d+[\.)]\s+")
    _unordered_markers = ("- ", "* ", "+ ")

    def process(self, data: ParseContext) -> ParseContext:
        data.blocks = [
            Block(
                type=self._block_type(data, index),
                content=self._block_content(data, index, paragraph),
            )
            for index, paragraph in enumerate(data.paragraphs)
        ]

        return data

    def _block_type(self, data: ParseContext, index: int) -> BlockType:
        if index in data.heading_indexes:
            return BlockType.HEADING

        if index in data.list_indexes:
            return BlockType.LIST

        return BlockType.TEXT

    def _block_content(
        self,
        data: ParseContext,
        index: int,
        paragraph: str,
    ) -> str:
        if index not in data.heading_indexes:
            if index in data.list_indexes:
                return self._clean_list(paragraph)

            return paragraph

        return paragraph.lstrip("#").strip()

    def _clean_list(self, paragraph: str) -> str:
        items = [
            self._clean_list_item(line.strip())
            for line in paragraph.splitlines()
            if line.strip()
        ]

        return "\n".join(items)

    def _clean_list_item(self, line: str) -> str:
        if line.startswith(self._unordered_markers):
            return line[2:].strip()

        return self._ordered_item_pattern.sub("", line).strip()
