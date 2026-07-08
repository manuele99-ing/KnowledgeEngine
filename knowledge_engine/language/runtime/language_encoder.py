from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort


class LanguageEncoder:
    """
    Encodes text into semantic embeddings using the
    KnowledgeEngine language model.

    This class is the single entry point for semantic
    encoding during runtime.

    It is intentionally independent from the retrieval
    module and from any specific LLM.
    """

    def __init__(
        self,
        model_path: Path,
    ) -> None:

        self._session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )

        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    def encode(
        self,
        text: str,
    ) -> np.ndarray:
        """
        Encodes a piece of text.

        Parameters
        ----------
        text
            Input sentence or document.

        Returns
        -------
        numpy.ndarray

            Embedding vector.
        """

        #
        # Placeholder.
        #
        # The preprocessing/tokenization pipeline will be
        # implemented once the first ONNX model is trained.
        #

        raise NotImplementedError(
            "Language encoder preprocessing "
            "has not been implemented yet."
        )

    @property
    def embedding_size(
        self,
    ) -> int:
        """
        Returns the output embedding dimension.

        Currently unavailable until the first
        model is exported.
        """

        raise NotImplementedError()