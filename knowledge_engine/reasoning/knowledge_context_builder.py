from __future__ import annotations

from knowledge_engine.domain.document.block import Block
from knowledge_engine.domain.document.document import Document


class KnowledgeContextBuilder:
    """
    Builds a compact context from the retrieved blocks.
    """

    def build(
        self,
        matches: list[tuple[Document, str, Block]],
    ) -> str:

        lines: list[str] = []

        for document, section_title, block in matches:

            content = block.content.strip()

            if not content:
                continue

            lines.append(
                f"Document: {document.title} ({document.source.name})"
            )
            lines.append(f"Section: {section_title}")
            lines.append(f"{block.type.name}: {content}")
            lines.append("")

        return "\n".join(lines).strip()