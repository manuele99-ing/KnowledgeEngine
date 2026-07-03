from __future__ import annotations


class LemmaNormalizer:
    """
    Normalizes Italian lemmas into a canonical lexical form.

    This component is intentionally independent from spaCy.
    It only transforms already-lemmatized tokens.
    """

    _NORMALIZATION_MAP = {
        # installazione
        "installazione": "installare",
        "installatore": "installare",

        # configurazione
        "configurazione": "configurare",
        "configuratore": "configurare",

        # collegamento
        "collegamento": "collegare",

        # connessione
        "connessione": "connettere",

        # attivazione
        "attivazione": "attivare",

        # creazione
        "creazione": "creare",

        # protezione
        "protezione": "proteggere",

        # salvataggio
        "salvataggio": "salvare",

        # registrazione
        "registrazione": "registrare",

        # alimentazione
        "alimentazione": "alimentare",

        # verifica
        "verifica": "verificare",

        # aggiornamento
        "aggiornamento": "aggiornare",
    }

    def normalize(
        self,
        lemma: str,
    ) -> str:

        return self._NORMALIZATION_MAP.get(
            lemma,
            lemma,
        )