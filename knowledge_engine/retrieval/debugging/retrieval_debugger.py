from __future__ import annotations

from knowledge_engine.retrieval.config.retrieval_config import RetrievalConfig
from knowledge_engine.retrieval.indexing.indexed_section import IndexedSection


class RetrievalDebugger:
    """
    Prints diagnostic information about retrieval results.

    This class is intentionally independent from the retrieval
    algorithm so it can be reused by BM25, FAISS or Hybrid
    retrievers.
    """

    def __init__(
        self,
        config: RetrievalConfig | None = None,
    ) -> None:

        self._config = config or RetrievalConfig()

    def print(
        self,
        *,
        question: str,
        query_tokens: list[str],
        ranked_sections: list[tuple[IndexedSection, float]],
    ) -> None:

        if not self._config.debug_enabled:
            return

        print()
        print("=" * 60)
        print("RETRIEVAL DEBUG")
        print("=" * 60)

        print(f"Question : {question}")
        print(f"Tokens   : {', '.join(query_tokens)}")
        print()

        for rank, (section, score) in enumerate(ranked_sections, start=1):

            print("-" * 60)
            print(f"Rank     : {rank}")
            print(f"Document : {section.document.source.name}")
            print(f"Section  : {' > '.join(section.path)}")
            print(f"Type     : {section.section.type.name}")
            print(f"Score    : {score:.2f}")
            print("Preview  :")
            print(self._build_preview(section))
            print()

    def _build_preview(
        self,
        section: IndexedSection,
    ) -> str:

        text = " ".join(section.text.split())

        if len(text) <= self._config.preview_length:
            return text

        return text[: self._config.preview_length].rstrip() + "..."