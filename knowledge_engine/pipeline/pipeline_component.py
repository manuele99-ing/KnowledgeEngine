from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class PipelineComponent(ABC):
    """
    Base class for every pipeline component.

    A component receives an input object, processes it,
    and returns the transformed output.
    """

    @abstractmethod
    def process(self, data: Any) -> Any:
        """
        Process the input object and return the result.
        """
        pass