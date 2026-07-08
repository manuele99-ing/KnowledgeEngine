from __future__ import annotations

from collections.abc import Iterable, Iterator


class DuplicateRemover:
    """
    Removes duplicated textual entries.

    The remover operates lazily and preserves the original order.

    Current implementation:

        - exact duplicate removal

    Future implementations may support:

        - fuzzy duplicates
        - MinHash
        - SimHash
        - semantic duplicates
        - embedding similarity
    """

    def remove(
        self,
        documents: Iterable[str],
    ) -> Iterator[str]:
        """
        Removes duplicated documents lazily.

        Parameters
        ----------
        documents
            Stream of textual documents.

        Yields
        ------
        str
            Unique documents preserving their original order.
        """

        seen: set[str] = set()

        for document in documents:

            normalized = self._normalize(
                document
            )

            if normalized in seen:
                continue

            seen.add(normalized)

            yield document

    @staticmethod
    def _normalize(
        text: str,
    ) -> str:
        """
        Normalizes text before duplicate comparison.
        """

        return " ".join(
            text.strip().lower().split()
        )