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
    def sections(self) -> tuple[IndexedSection, ...]:
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

        text_parts = [
            *section_path,
            *(
                block.content.strip()
                for block in section.blocks
                if block.content.strip()
            ),
        ]

        text = "\n".join(text_parts)

        indexed_sections.append(
            IndexedSection(
                document=document,
                section=section,
                text=text,
                tokens=tokenizer.tokenize(text),
                path=section_path,
                vector=cls._section_vector(tokenizer, text),
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