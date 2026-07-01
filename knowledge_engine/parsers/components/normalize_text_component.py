from __future__ import annotations

from dataclasses import replace

from knowledge_engine.domain.document.raw_document import RawDocument
from knowledge_engine.pipeline.pipeline_component import PipelineComponent


class NormalizeTextComponent(PipelineComponent):
    """
    Performs basic text normalization.
    """

    def process(self, data: RawDocument) -> RawDocument:
        lines = [line.rstrip() for line in data.content.splitlines()]

        content = "\n".join(lines).strip()

        return replace(
            data,
            content=content,
        )