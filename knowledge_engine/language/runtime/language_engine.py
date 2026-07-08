from __future__ import annotations

from pathlib import Path

from knowledge_engine.language.runtime.intent_detector import (
    Intent,
    IntentDetector,
)
from knowledge_engine.language.runtime.language_encoder import (
    LanguageEncoder,
)
from knowledge_engine.language.runtime.semantic_analyzer import (
    SemanticAnalyzer,
)
from knowledge_engine.language.runtime.semantic_query import (
    SemanticQuery,
)


class LanguageEngine:
    """
    Public entry point of the Language module.

    The rest of KnowledgeEngine should only interact
    with this class.

    The internal implementation (ONNX, HuggingFace,
    SentenceTransformers, tokenizer, etc.) remains
    completely hidden.

    Example
    -------

        language = LanguageEngine(model_path)

        semantic = language.analyze(query)
    """

    def __init__(
        self,
        model_path: Path,
    ) -> None:

        self._encoder = LanguageEncoder(
            model_path
        )

        self._intent_detector = IntentDetector(
            encoder=self._encoder,
            intents=self._default_intents(),
        )

        self._analyzer = SemanticAnalyzer(
            encoder=self._encoder,
            intent_detector=self._intent_detector,
        )

    def analyze(
        self,
        query: str,
    ) -> SemanticQuery:
        """
        Performs the complete semantic analysis of
        a user query.
        """

        return self._analyzer.analyze(
            query
        )

    @property
    def encoder(
        self,
    ) -> LanguageEncoder:
        """
        Exposes the encoder for future advanced
        semantic modules.

        Retrieval should normally never access it.
        """

        return self._encoder

    @property
    def analyzer(
        self,
    ) -> SemanticAnalyzer:

        return self._analyzer

    @staticmethod
    def _default_intents() -> list[Intent]:
        """
        Default semantic intents.

        Every intent is described using natural
        language instead of rules.

        The semantic similarity between the query
        and these descriptions determines the
        detected intent.
        """

        return [

            Intent(
                name="definition",
                description=(
                    "The user wants to know "
                    "what something is."
                ),
            ),

            Intent(
                name="description",
                description=(
                    "The user requests a general "
                    "description or explanation."
                ),
            ),

            Intent(
                name="installation",
                description=(
                    "The user asks how to install "
                    "or set up something."
                ),
            ),

            Intent(
                name="configuration",
                description=(
                    "The user wants to configure "
                    "or customize something."
                ),
            ),

            Intent(
                name="procedure",
                description=(
                    "The user asks how to perform "
                    "an operation."
                ),
            ),

            Intent(
                name="comparison",
                description=(
                    "The user wants to compare "
                    "two or more things."
                ),
            ),

            Intent(
                name="list",
                description=(
                    "The user asks for a list "
                    "of items."
                ),
            ),

            Intent(
                name="location",
                description=(
                    "The user asks where "
                    "something is located."
                ),
            ),

            Intent(
                name="summary",
                description=(
                    "The user wants a summary "
                    "or overview."
                ),
            ),

            Intent(
                name="error",
                description=(
                    "The user needs help solving "
                    "an error or problem."
                ),
            ),

        ]