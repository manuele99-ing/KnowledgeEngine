from __future__ import annotations

from collections.abc import Iterator
from typing import Any


class HuggingFaceParser:
    """
    Parses Hugging Face datasets.

    The parser attempts to automatically discover the most
    appropriate textual field.

    It is intentionally generic and dataset-independent.
    """

    #
    # Ordered by preference.
    #
    _TEXT_FIELDS = (
        "text",
        "content",
        "document",
        "body",
        "article",
        "paragraph",
        "sentence",
        "description",
        "abstract",
    )

    def parse(
        self,
        dataset: Any,
    ) -> Iterator[str]:
        """
        Lazily extracts textual documents.

        Parameters
        ----------
        dataset
            Hugging Face Dataset.

        Yields
        ------
        str
            One textual document at a time.
        """

        field = self._detect_text_field(dataset)

        if field is None:
            raise RuntimeError(
                "Unable to locate a textual field "
                "inside the Hugging Face dataset."
            )

        for row in dataset:

            value = row.get(field)

            if not isinstance(value, str):
                continue

            value = value.strip()

            if not value:
                continue

            yield value

    def _detect_text_field(
        self,
        dataset: Any,
    ) -> str | None:
        """
        Attempts to automatically detect
        the textual column.
        """

        if not hasattr(dataset, "column_names"):
            return None

        columns = set(dataset.column_names)

        #
        # Preferred names first.
        #
        for field in self._TEXT_FIELDS:

            if field in columns:
                return field

        #
        # Fallback:
        # first string column.
        #
        sample = dataset[0]

        for name, value in sample.items():

            if isinstance(value, str):
                return name

        return None