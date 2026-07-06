from __future__ import annotations

from enum import Enum, auto


class QueryIntent(Enum):
    """
    High-level intent detected from a user question.

    The retrieval pipeline uses this information to
    improve ranking without requiring an LLM.
    """

    UNKNOWN = auto()

    #
    # Knowledge
    #

    DEFINITION = auto()
    DESCRIPTION = auto()
    ROLE = auto()
    ENTITY = auto()

    #
    # Navigation
    #

    LOCATION = auto()

    #
    # Procedures
    #

    PROCEDURE = auto()
    INSTALLATION = auto()
    CONFIGURATION = auto()

    #
    # Lists
    #

    LIST = auto()
    COMPARISON = auto()

    #
    # Troubleshooting
    #

    ERROR = auto()

    #
    # Generic
    #

    SUMMARY = auto()