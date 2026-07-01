from __future__ import annotations

from typing import Any

from .pipeline_component import PipelineComponent


class Pipeline:
    """
    Executes a sequence of pipeline components.

    The output of each component becomes the input of the next one.
    """

    def __init__(self) -> None:
        self._components: list[PipelineComponent] = []

    def add_component(self, component: PipelineComponent) -> None:
        """
        Add a component to the pipeline.
        """
        self._components.append(component)

    def run(self, data: Any) -> Any:
        """
        Execute the pipeline.
        """
        for component in self._components:
            data = component.process(data)

        return data

    def clear(self) -> None:
        """
        Remove all components from the pipeline.
        """
        self._components.clear()

    @property
    def components(self) -> tuple[PipelineComponent, ...]:
        """
        Read-only view of the pipeline components.
        """
        return tuple(self._components)