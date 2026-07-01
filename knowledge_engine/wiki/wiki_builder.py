from __future__ import annotations

from knowledge_engine.domain.knowledge.knowledge_base import KnowledgeBase
from knowledge_engine.wiki.wiki import Wiki
from knowledge_engine.wiki.wiki_index import WikiIndex
from knowledge_engine.wiki.wiki_page import WikiPage


class WikiBuilder:
    """
    Builds an immutable Wiki from a KnowledgeBase.
    """

    def build(self, knowledge_base: KnowledgeBase) -> Wiki:

        pages: list[WikiPage] = []

        # Entity pages
        for entity in sorted(
            knowledge_base.entities,
            key=lambda entity: entity.name.casefold(),
        ):
            pages.append(
                WikiPage(
                    title=entity.name,
                    content=self._build_entity_page(
                        entity,
                        knowledge_base,
                    ),
                )
            )

        # Document pages
        for document in knowledge_base.documents:
            pages.append(
                WikiPage(
                    title=document.title,
                    content=self._build_document_page(document),
                )
            )

        index = WikiIndex(
            pages=tuple(page.title for page in pages),
        )

        return Wiki(
            index=index,
            pages=tuple(pages),
        )

    def _build_entity_page(
        self,
        entity,
        knowledge_base: KnowledgeBase,
    ) -> str:

        lines: list[str] = []

        lines.append(f"# {entity.name}")
        lines.append("")
        lines.append(f"Type: {entity.type.name}")
        lines.append("")

        relations = [
            relation
            for relation in knowledge_base.relations
            if relation.source == entity
            or relation.target == entity
        ]

        if relations:
            lines.append("## Relations")
            lines.append("")

            for relation in relations:
                lines.append(
                    f"- {relation.source.name} "
                    f"{relation.type.name} "
                    f"{relation.target.name}"
                )

        return "\n".join(lines).strip()

    def _build_document_page(
        self,
        document,
    ) -> str:

        lines: list[str] = []

        lines.append(f"# {document.title}")
        lines.append("")

        for section in document.sections:

            lines.append(f"## {section.title}")
            lines.append("")

            for block in section.blocks:
                lines.append(block.content)

            lines.append("")

        return "\n".join(lines).strip()