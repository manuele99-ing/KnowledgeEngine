from __future__ import annotations

from pathlib import Path

import fasttext


class LanguageFilter:
    """
    Filters documents according to their detected language.

    Language detection is performed using the fastText
    language identification model.

    The filter is completely offline.
    """

    def __init__(
        self,
        model_path: Path,
        target_language: str = "it",
        minimum_confidence: float = 0.80,
    ) -> None:

        self._model = fasttext.load_model(
            str(model_path)
        )

        self._target_language = target_language
        self._minimum_confidence = minimum_confidence

    def accepts(
        self,
        text: str,
    ) -> bool:
        """
        Returns True if the text belongs to the
        configured language.
        """

        if not text.strip():
            return False

        label, confidence = self._model.predict(
            text.replace("\n", " "),
            k=1,
        )

        language = label[0].replace("__label__", "")

        return (
            language == self._target_language
            and confidence[0] >= self._minimum_confidence
        )

    def detect(
        self,
        text: str,
    ) -> tuple[str, float]:
        """
        Returns the detected language together with
        its confidence.
        """

        label, confidence = self._model.predict(
            text.replace("\n", " "),
            k=1,
        )

        language = label[0].replace("__label__", "")

        return (
            language,
            float(confidence[0]),
        )