from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class Answer:
    """
    Answer produced by the reasoning engine.
    """

    question: str

    text: str

    context: str

    metadata: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
