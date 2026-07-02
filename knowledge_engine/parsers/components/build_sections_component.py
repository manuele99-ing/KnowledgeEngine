from __future__ import annotations

from dataclasses import dataclass, field

from knowledge_engine.domain.document.block import Block, BlockType
from knowledge_engine.domain.document.section import Section, SectionType
from knowledge_engine.parsers.context.parse_context import ParseContext
from knowledge_engine.pipeline.pipeline_component import PipelineComponent


@dataclass
class _MutableSection:
    title: str
    type: SectionType
    blocks: list[Block] = field(default_factory=list)
    children: list["_MutableSection"] = field(default_factory=list)
    level: int = 1


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

        root_sections: list[_MutableSection] = []
        stack: list[_MutableSection] = []
        intro_blocks: list[Block] = []

        for index, block in enumerate(data.blocks):
            paragraph = data.paragraphs[index]

            if paragraph.is_heading and block.type is BlockType.HEADING:
                section = _MutableSection(
                    title=block.content,
                    type=self._section_type(paragraph.heading_level),
                    level=paragraph.heading_level or 1,
                )
                self._attach_section(root_sections, stack, section)
                continue

            if stack:
                stack[-1].blocks.append(block)
            else:
                intro_blocks.append(block)

        if intro_blocks:
            root_sections.insert(
                0,
                _MutableSection(
                    title="Content",
                    type=SectionType.CHAPTER,
                    blocks=intro_blocks,
                    level=1,
                ),
            )

        data.sections = tuple(self._build_sections(root_sections))

        return data

    def _attach_section(
        self,
        root_sections: list[_MutableSection],
        stack: list[_MutableSection],
        section: _MutableSection,
    ) -> None:
        while stack and stack[-1].level >= section.level:
            stack.pop()

        if stack:
            stack[-1].children.append(section)
        else:
            root_sections.append(section)

        stack.append(section)

    def _build_sections(
        self,
        sections: list[_MutableSection],
    ) -> list[Section]:
        return [
            Section(
                title=section.title,
                type=section.type,
                blocks=tuple(section.blocks),
                children=tuple(self._build_sections(section.children)),
            )
            for section in sections
        ]

    def _section_type(self, level: int | None) -> SectionType:
        if level is None or level <= 1:
            return SectionType.CHAPTER

        if level == 2:
            return SectionType.SECTION

        return SectionType.SUBSECTION
