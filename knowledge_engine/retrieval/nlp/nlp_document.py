from __future__ import annotations

from dataclasses import dataclass

from spacy.tokens import Doc


@dataclass(frozen=True, slots=True)
class NLPDocument:
    """
    Result of a single spaCy parsing operation.

    The parsed document can expose multiple linguistic
    representations without parsing the same text again.
    """

    doc: Doc

    tokens: tuple[str, ...]

    vector: tuple[float, ...]