from __future__ import annotations

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

                text = "\n".join(
                    block.content.strip()
                    for block in section.blocks
                    if block.content.strip()
                )

                indexed_sections.append(
                    IndexedSection(
                        document=document,
                        section=section,
                        text=text,
                        tokens=tokenizer.tokenize(text),
                    )
                )

        return cls(tuple(indexed_sections))