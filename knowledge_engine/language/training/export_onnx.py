from __future__ import annotations

from pathlib import Path

from sentence_transformers import SentenceTransformer


class OnnxExporter:
    """
    Exports a trained SentenceTransformer model
    to ONNX format.

    The exported model is the one used at runtime
    by KnowledgeEngine.
    """

    def export(
        self,
        model_directory: Path,
        output_directory: Path,
    ) -> Path:
        """
        Exports the model to ONNX.

        The implementation will be completed once the
        first encoder has been successfully trained.
        """

        model = SentenceTransformer(
            str(model_directory)
        )

        raise NotImplementedError(
            "ONNX export has not yet been implemented."
        )