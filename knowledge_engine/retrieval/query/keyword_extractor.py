from __future__ import annotations

from spacy.tokens import Doc, Token


class KeywordExtractor:
    """
    Extracts the most informative keywords from a parsed query.

    The extraction is completely domain-independent and relies
    on linguistic information produced by spaCy.

    Returned keywords are ordered by importance and already
    lemmatized by spaCy.
    """

    #
    # Universal POS tags considered informative.
    #

    _CONTENT_POS = {
        "NOUN",
        "PROPN",
        "VERB",
        "ADJ",
        "NUM",
    }

    #
    # Dependency labels that usually identify
    # the semantic focus of a sentence.
    #

    _IMPORTANT_DEPENDENCIES = {
        "ROOT",
        "nsubj",
        "obj",
        "obl",
        "attr",
        "appos",
        "compound",
        "conj",
    }

    def extract(
        self,
        doc: Doc,
    ) -> tuple[str, ...]:
        """
        Extracts ranked keywords from a parsed document.
        """

        scored_tokens: list[tuple[int, str]] = []

        for token in doc:

            if not self._is_candidate(token):
                continue

            score = self._score(token)

            lemma = token.lemma_.strip().lower()

            if not lemma:
                continue

            scored_tokens.append(
                (
                    score,
                    lemma,
                )
            )

        scored_tokens.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        keywords: list[str] = []
        seen: set[str] = set()

        for _, lemma in scored_tokens:

            if lemma in seen:
                continue

            seen.add(lemma)
            keywords.append(lemma)

        return tuple(keywords)

    def _is_candidate(
        self,
        token: Token,
    ) -> bool:

        if token.is_space:
            return False

        if token.is_punct:
            return False

        if token.is_stop:
            return False

        if token.like_url:
            return False

        if token.like_email:
            return False

        if token.pos_ not in self._CONTENT_POS:
            return False

        return True

    def _score(
        self,
        token: Token,
    ) -> int:

        score = 0

        #
        # Proper names are usually the most important
        # terms in information retrieval.
        #

        if token.pos_ == "PROPN":
            score += 100

        #
        # Sentence root.
        #

        if token.dep_ == "ROOT":
            score += 40

        #
        # Subject / object.
        #

        if token.dep_ in self._IMPORTANT_DEPENDENCIES:
            score += 20

        #
        # Nouns generally carry more semantics
        # than verbs.
        #

        if token.pos_ == "NOUN":
            score += 15

        if token.pos_ == "VERB":
            score += 10

        if token.pos_ == "ADJ":
            score += 5

        return score