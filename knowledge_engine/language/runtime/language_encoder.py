from __future__ import annotations

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


class LanguageEncoder:
    """
    Encodes text into semantic embeddings using the
    KnowledgeEngine language model.

    This implementation currently uses a
    SentenceTransformer model (BGE).

    The rest of the project remains completely
    independent from the underlying implementation,
    making it possible to switch to ONNX later
    without changing Retrieval.
    """

    def __init__(
        self,
        model_path: Path | str,
    ) -> None:

        if isinstance(model_path, Path):
            model_path = str(model_path)

        self._model = SentenceTransformer(
            model_path,
            trust_remote_code=False,
        )

    def encode(
        self,
        text: str,
    ) -> np.ndarray:
        """
        Encodes a text into a normalized embedding.

        Parameters
        ----------
        text
            Input sentence or document.

        Returns
        -------
        numpy.ndarray
            Normalized embedding vector.
        """

        embedding = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embedding.astype(np.float32)

    @property
    def embedding_size(
        self,
    ) -> int:
        """
        Returns the embedding dimension.
        """

        return int(
            self._model.get_embedding_dimension()
        )