from __future__ import annotations

import html
import re


class Cleaner:
    """
    Performs generic text cleaning.

    The cleaner is intentionally dataset-independent.

    It normalizes text before sentence splitting and
    language processing.

    Operations include:

        - HTML entity decoding
        - whitespace normalization
        - control character removal
        - empty line removal
    """

    _MULTIPLE_SPACES = re.compile(r"\s+")

    _CONTROL_CHARS = re.compile(
        r"[\x00-\x08\x0B\x0C\x0E-\x1F]"
    )

    def clean(
        self,
        text: str,
    ) -> str:
        """
        Cleans raw text.

        Parameters
        ----------
        text
            Raw document.

        Returns
        -------
        Cleaned document.
        """

        text = html.unescape(text)

        text = self._CONTROL_CHARS.sub(
            "",
            text,
        )

        text = text.replace("\r", "\n")

        text = self._normalize_blank_lines(text)

        text = self._MULTIPLE_SPACES.sub(
            " ",
            text,
        )

        return text.strip()

    @staticmethod
    def _normalize_blank_lines(
        text: str,
    ) -> str:

        lines = []

        previous_blank = False

        for line in text.split("\n"):

            line = line.strip()

            if not line:

                if previous_blank:
                    continue

                previous_blank = True
                lines.append("")
                continue

            previous_blank = False
            lines.append(line)

        return "\n".join(lines)