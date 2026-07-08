from __future__ import annotations

from pathlib import Path

from sentence_transformers import SentenceTransformer


class EncoderTrainer:
    """
    Trains the KnowledgeEngine language encoder.

    The trainer orchestrates the complete training process
    but does not implement any model-specific logic.

    A preprocessed corpus is used as input.

    The resulting model is exported in SentenceTransformer
    format and can later be converted to ONNX.
    """

    def __init__(
        self,
        base_model: str,
        output_directory: Path,
    ) -> None:

        self._base_model = base_model
        self._output_directory = output_directory

        self._model = SentenceTransformer(
            base_model
        )

    @property
    def model(self) -> SentenceTransformer:

        return self._model

    def train(
        self,
        corpus,
    ) -> None:
        """
        Trains the encoder.

        Current implementation is intentionally left as a
        placeholder until the training dataset pipeline
        is completed.
        """

        raise NotImplementedError(
            "Training pipeline has not yet been implemented."
        )

    def save(
        self,
    ) -> Path:
        """
        Saves the trained model.
        """

        self._model.save(
            str(
                self._output_directory
            )
        )

        return self._output_directory