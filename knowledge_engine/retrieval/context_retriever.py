from __future__ import annotations

from knowledge_engine.domain.document.block import Block
from knowledge_engine.domain.document.document import Document
from knowledge_engine.domain.knowledge.knowledge_base import KnowledgeBase


class ContextRetriever:
    """
    Retrieves the most relevant document blocks for a question.

    Current implementation:
    - simple keyword matching

    Future implementations may use:
    - TF-IDF
    - BM25
    - Embeddings
    - Vector databases
    """

    def retrieve(
        self,
        question: str,
        knowledge_base: KnowledgeBase,
    ) -> list[tuple[Document, str, Block]]:
        keywords = {
            word.lower()
            for word in question.split()
            if len(word) >= 3
        }

        matches: list[tuple[int, Document, str, Block]] = []

        for document in knowledge_base.documents:
            for section in document.sections:
                for block in section.blocks:

                    text = block.content.lower()

                    score = sum(
                        keyword in text
                        for keyword in keywords
                    )

                    if score == 0:
                        continue

                    matches.append(
                        (
                            score,
                            document,
                            section.title,
                            block,
                        )
                    )

        matches.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            (document, section_title, block)
            for _, document, section_title, block in matches[:10]
        ]