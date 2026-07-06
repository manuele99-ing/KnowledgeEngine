from __future__ import annotations

from knowledge_engine.domain.document.document import Document
from knowledge_engine.domain.document.section import Section
from knowledge_engine.domain.knowledge.knowledge_base import KnowledgeBase
from knowledge_engine.retrieval.indexing.indexed_section import IndexedSection
from knowledge_engine.retrieval.nlp.spacy_tokenizer import SpacyTokenizer


class SectionIndex:
    """
    Search index built from a KnowledgeBase.
    """

    def __init__(
        self,
        sections: tuple[IndexedSection, ...],
    ) -> None:
        self._sections = sections

    @property
    def sections(
        self,
    ) -> tuple[IndexedSection, ...]:
        return self._sections

    @classmethod
    def build(
        cls,
        knowledge_base: KnowledgeBase,
        tokenizer: SpacyTokenizer | None = None,
    ) -> "SectionIndex":

        tokenizer = tokenizer or SpacyTokenizer()

        indexed_sections: list[IndexedSection] = []

        for document in knowledge_base.documents:
            for section in document.sections:
                cls._index_section(
                    document=document,
                    section=section,
                    tokenizer=tokenizer,
                    path=(),
                    indexed_sections=indexed_sections,
                )

        return cls(tuple(indexed_sections))

    @staticmethod
    def _section_vector(
        tokenizer: SpacyTokenizer,
        text: str,
    ) -> tuple[float, ...]:

        return tokenizer.vector(text)

    @classmethod
    def _index_section(
        cls,
        document: Document,
        section: Section,
        tokenizer: SpacyTokenizer,
        path: tuple[str, ...],
        indexed_sections: list[IndexedSection],
    ) -> None:

        section_path = (*path, section.title)

        title = section.title.strip()

        body_parts = [
            block.content.strip()
            for block in section.blocks
            if block.content.strip()
        ]

        body = "\n".join(body_parts)

        full_text = "\n".join(
            [
                *section_path,
                body,
            ]
        ).strip()

        title_tokens = tokenizer.tokenize(title)

        body_tokens = tokenizer.tokenize(body)

        path_tokens = tokenizer.tokenize(" ".join(section_path))

        tokens = (
            title_tokens
            + path_tokens
            + body_tokens
        )

        indexed_sections.append(
            IndexedSection(
                document=document,
                section=section,
                title=title,
                body=body,
                text=full_text,
                title_tokens=title_tokens,
                body_tokens=body_tokens,
                tokens=tokens,
                path=section_path,
                vector=cls._section_vector(
                    tokenizer,
                    full_text,
                ),
            )
        )

        for child in section.children:
            cls._index_section(
                document=document,
                section=child,
                tokenizer=tokenizer,
                path=section_path,
                indexed_sections=indexed_sections,
            )