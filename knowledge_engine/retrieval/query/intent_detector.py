from __future__ import annotations

from spacy.tokens import Doc

from knowledge_engine.retrieval.query.query_intent import QueryIntent


class IntentDetector:
    """
    Detects the user's intent from a parsed Italian question.

    The detector is language-oriented rather than domain-oriented.
    It relies on grammatical structure, dependency parsing and
    interrogative pronouns instead of domain-specific keywords.
    """

    def detect(
        self,
        doc: Doc,
    ) -> QueryIntent:

        if len(doc) == 0:
            return QueryIntent.UNKNOWN

        first = self._first_content_token(doc)

        if first is None:
            return QueryIntent.UNKNOWN

        first = first.lower()

        #
        # Definition
        #
        # Chi è Elsa?
        #

        if first == "chi":
            return QueryIntent.DEFINITION

        #
        # Location
        #
        # Dove si trova...
        #

        if first == "dove":
            return QueryIntent.LOCATION

        #
        # Procedure
        #
        # Come installare...
        #

        if first == "come":

            root = self._root(doc)

            if root is not None and root.pos_ == "VERB":
                return QueryIntent.PROCEDURE

            return QueryIntent.UNKNOWN

        #
        # Lists
        #
        # Quali dispositivi...
        #

        if first in {
            "quale",
            "quali",
        }:
            return QueryIntent.LIST

        #
        # Generic description.
        #

        if first == "cosa":
            return QueryIntent.DESCRIPTION

        return QueryIntent.UNKNOWN

    @staticmethod
    def _first_content_token(
        doc: Doc,
    ) -> str | None:

        for token in doc:

            if token.is_space:
                continue

            if token.is_punct:
                continue

            return token.lemma_.strip().lower()

        return None

    @staticmethod
    def _root(
        doc: Doc,
    ):

        for token in doc:

            if token.dep_ == "ROOT":
                return token

        return None