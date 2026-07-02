from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """
    Configuration for the retrieval subsystem.
    """

    debug_enabled: bool = False

    preview_length: int = 180

    default_top_k: int = 3

    bm25_k1: float = 1.5

    bm25_b: float = 0.0

    semantic_weight: float = 0.5