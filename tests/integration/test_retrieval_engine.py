from __future__ import annotations

import unittest

from knowledge_engine.loaders.directory_loader import DirectoryLoader
from knowledge_engine.parsers.text_parser import TextParser
from knowledge_engine.semantic.knowledge_extractor import KnowledgeExtractor
from knowledge_engine.retrieval.retrieval_engine import RetrievalEngine


TEST_CASES = [

    (
        "Come si installa il dispositivo?",
        "installazione.txt.txt",
        "Requisiti",
    ),

    (
        "Come collego il dispositivo?",
        "installazione.txt.txt",
        "Collegamento",
    ),

    (
        "Come verifico il dispositivo?",
        "installazione.txt.txt",
        "Verifica",
    ),

    (
        "Come imposto un IP statico?",
        "configurazione.txt.txt",
        "Rete",
    ),

    (
        "Password di default",
        "configurazione.txt.txt",
        "Accesso",
    ),

    (
        "Come aggiorno il firmware?",
        "manutenzione.txt.txt",
        "Aggiornamento firmware",
    ),
]


class RetrievalEngineIntegrationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        loader = DirectoryLoader()
        parser = TextParser()
        extractor = KnowledgeExtractor()

        raw_documents = loader.load_directory("input")

        documents = [
            parser.parse(raw_document)
            for raw_document in raw_documents
        ]

        knowledge_base = extractor.extract(documents)

        cls.retrieval_engine = RetrievalEngine(
            knowledge_base=knowledge_base
        )

    def test_retrieval_cases(self) -> None:

        for (
            question,
            expected_document,
            expected_section,
        ) in TEST_CASES:

            with self.subTest(question=question):

                results = self.retrieval_engine.search(
                    question=question,
                    top_k=1,
                )

                self.assertGreater(
                    len(results),
                    0,
                    f"Nessun risultato per '{question}'",
                )

                result = results[0]

                self.assertEqual(
                    result.document.source.name,
                    expected_document,
                    (
                        f"\nDomanda : {question}"
                        f"\nDocumento atteso : {expected_document}"
                        f"\nDocumento ottenuto : {result.document.source.name}"
                    ),
                )

                self.assertEqual(
                    result.section.title,
                    expected_section,
                    (
                        f"\nDomanda : {question}"
                        f"\nSezione attesa : {expected_section}"
                        f"\nSezione ottenuta : {result.section.title}"
                    ),
                )


if __name__ == "__main__":
    unittest.main()