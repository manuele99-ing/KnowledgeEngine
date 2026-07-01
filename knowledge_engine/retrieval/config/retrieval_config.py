from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """
    Configuration for the retrieval subsystem.
    """

    debug_enabled: bool = False

    preview_length: int = 180

    default_top_k: int = 5