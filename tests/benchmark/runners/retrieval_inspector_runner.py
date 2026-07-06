from __future__ import annotations

import json
from pathlib import Path

from knowledge_engine.loaders.directory_loader import DirectoryLoader
from knowledge_engine.parsers.text_parser import TextParser
from knowledge_engine.retrieval.retrieval_engine import RetrievalEngine
from knowledge_engine.semantic.knowledge_extractor import KnowledgeExtractor


class RetrievalInspectorRunner:

    def __init__(
        self,
        dataset_path: str,
        input_directory: str = "input",
    ) -> None:

        self._dataset_path = Path(dataset_path)
        self._input_directory = input_directory

    def run(
        self,
    ) -> None:

        dataset = self._load_dataset()

        retrieval = self._build_retrieval_engine()

        total_tests = 0
        score = 0

        document_matches = 0
        section_matches = 0
        acceptable_matches = 0

        print()
        print("=" * 70)
        print(" RETRIEVAL INSPECTOR")
        print("=" * 70)

        for test in dataset:

            total_tests += 1

            print()
            print("-" * 70)

            print(f"TEST      : {test['id']}")
            print(f"NAME      : {test['name']}")
            print(f"PURPOSE   : {test['purpose']}")
            print(f"QUESTION  : {test['question']}")
            print()

            results = retrieval.search(
                test["question"],
            )

            if not results:

                print("BEST RESULT")
                print("  No sections retrieved")

                if test["expected_document"] is None:
                    score += 10

                print()
                continue

            best = results[0]

            print("BEST RESULT")
            print(f"  Document : {best.document.title}")
            print(f"  Section  : {best.section.title}")
            print(f"  Score    : {best.score:.3f}")

            print()
            print("EXPECTED")
            print(f"  Document : {test['expected_document']}")
            print(f"  Section  : {test['expected_section']}")

            document_ok = (
                best.document.title
                == test["expected_document"]
            )

            section_ok = (
                best.section.title
                == test["expected_section"]
            )

            acceptable_ok = (
                best.section.title
                in test.get(
                    "acceptable_sections",
                    [],
                )
            )

            if document_ok:
                document_matches += 1

            if section_ok:
                section_matches += 1

            if acceptable_ok:
                acceptable_matches += 1

            #
            # Inspector Score
            #

            if section_ok:

                score += 10

            elif acceptable_ok:

                score += 7

            elif document_ok:

                score += 3

            print()
            print("CHECKS")

            print(
                f"  Document Match : {'PASS' if document_ok else 'FAIL'}"
            )

            print(
                f"  Section Match  : {'PASS' if section_ok else 'FAIL'}"
            )

            print(
                f"  Acceptable     : {'YES' if acceptable_ok else 'NO'}"
            )

            print()
            print("TOP RESULTS")

            for index, result in enumerate(
                results[:5],
                start=1,
            ):

                print(
                    f"{index:>2}. "
                    f"{result.document.title} | "
                    f"{result.section.title} | "
                    f"{result.score:.3f}"
                )

        #
        # SUMMARY
        #

        print()
        print("=" * 70)
        print(" SUMMARY")
        print("=" * 70)

        document_accuracy = (
            document_matches / total_tests * 100
            if total_tests
            else 0.0
        )

        section_accuracy = (
            section_matches / total_tests * 100
            if total_tests
            else 0.0
        )

        acceptable_accuracy = (
            acceptable_matches / total_tests * 100
            if total_tests
            else 0.0
        )

        print(f"Tests                 : {total_tests}")
        print(f"Document Accuracy     : {document_accuracy:6.2f}%")
        print(f"Section Accuracy      : {section_accuracy:6.2f}%")
        print(f"Acceptable Sections   : {acceptable_accuracy:6.2f}%")

        print()
        print("=" * 70)
        print(f" RETRIEVAL SCORE      : {score:3d}/100")
        print("=" * 70)

    def _load_dataset(
        self,
    ):

        with self._dataset_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    def _build_knowledge_base(
        self,
    ):

        loader = DirectoryLoader()

        parser = TextParser()

        extractor = KnowledgeExtractor()

        raw_documents = loader.load_directory(
            self._input_directory,
        )

        documents = [
            parser.parse(document)
            for document in raw_documents
        ]

        return extractor.extract(
            documents,
        )

    def _build_retrieval_engine(
        self,
    ):

        return RetrievalEngine(
            knowledge_base=self._build_knowledge_base(),
        )


def main():

    RetrievalInspectorRunner(
        dataset_path="tests/benchmark/datasets/retrieval_inspector.json",
        input_directory="input",
    ).run()


if __name__ == "__main__":
    main()