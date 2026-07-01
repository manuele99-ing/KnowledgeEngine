from __future__ import annotations

from knowledge_engine.domain.document.block import Block
from knowledge_engine.domain.knowledge.knowledge_base import KnowledgeBase


class KnowledgeContextBuilder:
    """
    Builds a compact text context from a KnowledgeBase.
    """

    def build(self, knowledge_base: KnowledgeBase) -> str:
        lines: list[str] = []

        lines.append("Documents:")
        for document in knowledge_base.documents:
            lines.append(f"- {document.title} ({document.source.name})")
            for section in document.sections:
                self._append_section(
                    lines=lines,
                    section_title=section.title,
                    blocks=section.blocks,
                )

        lines.append("")
        lines.append("Entities:")
        for entity in knowledge_base.entities:
            lines.append(f"- {entity.name} [{entity.type.name}]")

        lines.append("")
        lines.append("Relations:")
        for relation in knowledge_base.relations:
            lines.append(
                "- "
                f"{relation.source.name} "
                f"{relation.type.name} "
                f"{relation.target.name}"
            )

        return "\n".join(lines).strip()

    def _append_section(
        self,
        lines: list[str],
        section_title: str,
        blocks: tuple[Block, ...],
    ) -> None:
        lines.append(f"  Section: {section_title}")

        for block in blocks:
            content = block.content.strip()

            if not content:
                continue

            lines.append(f"  - {block.type.name}: {content}")
