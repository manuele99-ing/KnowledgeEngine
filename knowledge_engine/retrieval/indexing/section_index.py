from __future__ import annotations

from knowledge_engine.domain.document.document import Document
from knowledge_engine.domain.document.section import Section
from knowledge_engine.domain.knowledge.knowledge_base import KnowledgeBase
from knowledge_engine.language.runtime.language_engine import LanguageEngine
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
        language: LanguageEngine | None = None,
    ) -> "SectionIndex":

        tokenizer = tokenizer or SpacyTokenizer()

        indexed_sections: list[IndexedSection] = []

        for document in knowledge_base.documents:
            for section in document.sections:
                cls._index_section(
                    document=document,
                    section=section,
                    tokenizer=tokenizer,
                    language=language,
                    path=(),
                    indexed_sections=indexed_sections,
                )

        return cls(tuple(indexed_sections))

    @classmethod
    def _index_section(
        cls,
        document: Document,
        section: Section,
        tokenizer: SpacyTokenizer,
        language: LanguageEngine | None,
        path: tuple[str, ...],
        indexed_sections: list[IndexedSection],
    ) -> None:

        section_path = (*path, section.title)

        title = section.title.strip()

        #
        # Corpo della sezione corrente
        #

        body_parts = [
            block.content.strip()
            for block in section.blocks
            if block.content.strip()
        ]

        body = "\n".join(body_parts)

        #
        # Raccoglie ricorsivamente il testo
        # di tutte le sottosezioni.
        #

        descendants: list[str] = []

        def collect_children(
            current: Section,
        ) -> None:

            for child in current.children:

                if child.title.strip():
                    descendants.append(
                        child.title.strip()
                    )

                for block in child.blocks:

                    content = block.content.strip()

                    if content:
                        descendants.append(
                            content
                        )

                collect_children(child)

        collect_children(section)

        descendants_text = "\n".join(descendants)

        #
        # Testo completo della sezione.
        #

        full_text = "\n".join(
            [
                *section_path,
                body,
                descendants_text,
            ]
        ).strip()

        #
        # Tokenizzazione (spaCy)
        #

        title_tokens = tokenizer.tokenize(
            title
        )

        body_tokens = tokenizer.tokenize(
            body
        )

        descendants_tokens = tokenizer.tokenize(
            descendants_text
        )

        path_tokens = tokenizer.tokenize(
            " ".join(section_path)
        )

        tokens = (
            title_tokens
            + path_tokens
            + body_tokens
            + descendants_tokens
        )

        #
        # Semantic embedding (Language)
        #

        embedding = None

        if (
            language is not None
            and full_text
        ):
            embedding = language.encoder.encode(
                full_text
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
                embedding=embedding,
            )
        )

        #
        # Indicizza anche le sottosezioni.
        #

        for child in section.children:

            cls._index_section(
                document=document,
                section=child,
                tokenizer=tokenizer,
                language=language,
                path=section_path,
                indexed_sections=indexed_sections,
            )