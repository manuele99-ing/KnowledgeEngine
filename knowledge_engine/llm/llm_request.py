from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """
    Request sent to an LLM backend.
    """

    prompt: str

    metadata: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
