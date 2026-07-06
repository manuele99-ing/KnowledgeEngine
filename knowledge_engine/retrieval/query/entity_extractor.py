from __future__ import annotations

from spacy.tokens import Doc


class EntityExtractor:
    """
    Extracts entities from a parsed query.

    The extractor is completely domain-independent and relies on
    spaCy's Named Entity Recognition.

    If no entities are detected, it falls back to proper nouns,
    which generally represent the focus of user questions.
    """

    def extract(
        self,
        doc: Doc,
    ) -> tuple[str, ...]:

        entities: list[str] = []
        seen: set[str] = set()

        #
        # Primary extraction:
        # Named Entity Recognition.
        #

        for entity in doc.ents:

            value = entity.text.strip()

            if not value:
                continue

            key = value.lower()

            if key in seen:
                continue

            seen.add(key)
            entities.append(value)

        #
        # Fallback:
        # Proper nouns.
        #

        if not entities:

            for token in doc:

                if token.pos_ != "PROPN":
                    continue

                value = token.text.strip()

                if not value:
                    continue

                key = value.lower()

                if key in seen:
                    continue

                seen.add(key)
                entities.append(value)

        return tuple(entities)