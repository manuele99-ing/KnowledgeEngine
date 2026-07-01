from __future__ import annotations

from knowledge_engine.domain.document.block import Block, BlockType
from knowledge_engine.domain.document.section import Section, SectionType
from knowledge_engine.parsers.context.parse_context import ParseContext
from knowledge_engine.pipeline.pipeline_component import PipelineComponent


class BuildSectionsComponent(PipelineComponent):
    """
    Builds the initial section tree for a plain text document.
    """

    def process(self, data: ParseContext) -> ParseContext:
        if not data.blocks:
            data.sections = [
                Section(
                    title="Content",
                    type=SectionType.CHAPTER,
                )
            ]
            return data

        sections: list[Section] = []
        current_title = "Content"
        current_blocks: list[Block] = []

        for block in data.blocks:
            if block.type is BlockType.HEADING:
                self._append_section(
                    sections=sections,
                    title=current_title,
                    blocks=current_blocks,
                )
                current_title = block.content
                current_blocks = []
                continue

            current_blocks.append(block)

        self._append_section(
            sections=sections,
            title=current_title,
            blocks=current_blocks,
            include_empty=not sections,
        )

        data.sections = sections

        return data

    def _append_section(
        self,
        sections: list[Section],
        title: str,
        blocks: list[Block],
        include_empty: bool = False,
    ) -> None:
        if not blocks and not include_empty:
            return

        sections.append(
            Section(
                title=title,
                type=SectionType.CHAPTER,
                blocks=tuple(blocks),
            )
        )
