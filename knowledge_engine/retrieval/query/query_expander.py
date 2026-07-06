from __future__ import annotations

from spacy.tokens import Doc


class QueryExpander:
    """
    Expands a parsed query using linguistic information.

    The objective is not to invent new knowledge but to expose
    additional lexical forms already present in the query.

    Future implementations may integrate:

    - MultiWordNet
    - BabelNet
    - ItalWordNet
    - Domain thesauri

    without changing the retrieval pipeline.
    """

    def expand(
        self,
        doc: Doc,
    ) -> tuple[str, ...]:

        expanded: list[str] = []
        seen: set[str] = set()

        #
        # Lemmas
        #

        for token in doc:

            if token.is_space:
                continue

            if token.is_punct:
                continue

            if token.is_stop:
                continue

            lemma = token.lemma_.strip().lower()

            if lemma and lemma not in seen:
                seen.add(lemma)
                expanded.append(lemma)

        #
        # Compound expressions
        #

        for chunk in doc.noun_chunks:

            text = chunk.lemma_.strip().lower()

            if not text:
                continue

            if text in seen:
                continue

            seen.add(text)
            expanded.append(text)

        #
        # Named entities
        #

        for entity in doc.ents:

            text = entity.lemma_.strip().lower()

            if not text:
                text = entity.text.strip().lower()

            if not text:
                continue

            if text in seen:
                continue

            seen.add(text)
            expanded.append(text)

        return tuple(expanded)