from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class WikiIndex:
    """
    Represents the main index of the generated wiki.

    The index acts as the entry point of the wiki and contains
    references to all generated pages.
    """

    title: str = "Knowledge Engine Wiki"

    pages: tuple[str, ...] = field(default_factory=tuple)