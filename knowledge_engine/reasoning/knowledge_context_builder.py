from __future__ import annotations

from knowledge_engine.retrieval.retrieved_section import RetrievedSection


class KnowledgeContextBuilder:
    """
    Builds a compact context from retrieved sections.
    """

    def build(
        self,
        sections: list[RetrievedSection],
    ) -> str:

        lines: list[str] = []

        for item in sections:
            lines.append(
                f"Document: {item.document.title} ({item.document.source.name})"
            )
            lines.append(f"Section: {item.section.title}")
            lines.append("Content:")

            for block in item.section.blocks:
                content = block.content.strip()

                if not content:
                    continue

                lines.append(content)

            lines.append("")

        return "\n".join(lines).strip()