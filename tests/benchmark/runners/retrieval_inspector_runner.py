from __future__ import annotations

import json
from collections import defaultdict
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

    # ---------------------------------------------------------

    def run(self) -> None:

        dataset = self._load_dataset()

        retrieval = self._build_retrieval_engine()

        total_tests = 0
        score = 0

        document_matches = 0
        section_matches = 0
        acceptable_matches = 0

        purpose_stats = defaultdict(
            lambda: {
                "total": 0,
                "passed": 0,
            }
        )

        print()
        print("=" * 70)
        print(" RETRIEVAL INSPECTOR")
        print("=" * 70)

        for test in dataset:

            total_tests += 1

            purpose_stats[test["purpose"]]["total"] += 1

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

            expected_document = test.get(
                "expected_document"
            )

            expected_section = test.get(
                "expected_section"
            )

            #
            # NEGATIVE TEST
            #

            if expected_document is None:

                if not results:

                    print("BEST RESULT")
                    print("  No sections retrieved")

                    print()
                    print("CHECKS")
                    print("  Negative Retrieval : PASS")

                    score += 10
                    purpose_stats[test["purpose"]]["passed"] += 1

                    continue

                print("BEST RESULT")
                print(
                    f"  Document : {results[0].document.title}"
                )
                print(
                    f"  Section  : {results[0].section.title}"
                )
                print(
                    f"  Score    : {results[0].score:.3f}"
                )

                print()
                print("EXPECTED")
                print("  No result")

                print()
                print("FAILURE ANALYSIS")
                print(
                    "  Retrieval returned a result "
                    "when no answer should exist."
                )

                continue

            #
            # NO RESULTS
            #

            if not results:

                print("BEST RESULT")
                print("  No sections retrieved")

                print()
                print("EXPECTED")
                print(f"  Document : {expected_document}")
                print(f"  Section  : {expected_section}")

                print()
                print("FAILURE ANALYSIS")
                print("  Nothing was retrieved.")

                continue

            best = results[0]

            print("BEST RESULT")
            print(
                f"  Document : {best.document.title}"
            )
            print(
                f"  Section  : {best.section.title}"
            )
            print(
                f"  Score    : {best.score:.3f}"
            )

            print()
            print("EXPECTED")
            print(f"  Document : {expected_document}")
            print(f"  Section  : {expected_section}")

            #
            # DOCUMENT MATCH
            #

            document_ok = self._document_matches(
                best.document,
                expected_document,
            )

            section_ok = (
                best.section.title
                == expected_section
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

            if section_ok:
                purpose_stats[test["purpose"]]["passed"] += 1

            #
            # Inspector score
            #

            if section_ok:

                score += 10

            elif acceptable_ok:

                score += 7

            elif document_ok:

                score += 3

            #
            # Expected Rank
            #

            expected_rank = None
            expected_result = None

            for rank, result in enumerate(
                results,
                start=1,
            ):

                if (
                    self._document_matches(
                        result.document,
                        expected_document,
                    )
                    and result.section.title == expected_section
                ):

                    expected_rank = rank
                    expected_result = result
                    break

            #
            # Checks
            #

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

            if expected_rank is None:

                print("  Expected Rank  : NOT FOUND")

            else:

                print(
                    f"  Expected Rank  : {expected_rank}"
                )

            #
            # Score distance
            #

            if expected_result is not None:

                delta = (
                    best.score
                    - expected_result.score
                )

                print(
                    f"  Score Delta    : {delta:.3f}"
                )

            #
            # Failure Analysis
            #

            print()
            print("FAILURE ANALYSIS")

            if section_ok:

                print(
                    "  Retrieval selected the expected section."
                )

            else:

                reason = self._guess_failure_reason(
                    test["purpose"],
                    expected_rank,
                    acceptable_ok,
                )

                print(f"  {reason}")

            #
            # Future ranking breakdown
            #

            print()
            print("RANKING BREAKDOWN")

            print(
                "  Title Score  : N/A"
            )
            print(
                "  Body Score   : N/A"
            )
            print(
                "  Path Score   : N/A"
            )
            print(
                "  Entity Bonus : N/A"
            )
            print(
                "  Final Score  : "
                f"{best.score:.3f}"
            )

            #
            # Top results
            #

            print()
            print("TOP RESULTS")

            for index, result in enumerate(
                results[:5],
                start=1,
            ):

                marker = " "

                if (
                    self._document_matches(
                        result.document,
                        expected_document,
                    )
                    and result.section.title
                    == expected_section
                ):

                    marker = "*"

                print(
                    f"{marker}{index:>2}. "
                    f"{result.document.title} | "
                    f"{result.section.title} | "
                    f"{result.score:.3f}"
                )

        #
        # Summary
        #

        print()
        print("=" * 70)
        print(" SUMMARY")
        print("=" * 70)

        document_accuracy = (
            document_matches
            / total_tests
            * 100
            if total_tests
            else 0.0
        )

        section_accuracy = (
            section_matches
            / total_tests
            * 100
            if total_tests
            else 0.0
        )

        acceptable_accuracy = (
            acceptable_matches
            / total_tests
            * 100
            if total_tests
            else 0.0
        )

        print(
            f"Tests                 : {total_tests}"
        )

        print(
            f"Document Accuracy     : {document_accuracy:6.2f}%"
        )

        print(
            f"Section Accuracy      : {section_accuracy:6.2f}%"
        )

        print(
            f"Acceptable Sections   : {acceptable_accuracy:6.2f}%"
        )

        print()

        print("=" * 70)
        print(" PURPOSE BREAKDOWN")
        print("=" * 70)

        for purpose in sorted(
            purpose_stats.keys()
        ):

            total = purpose_stats[purpose]["total"]
            passed = purpose_stats[purpose]["passed"]

            accuracy = (
                passed / total * 100
                if total
                else 0
            )

            print(
                f"{purpose:<24}"
                f"{passed:>3}/{total:<3}"
                f" {accuracy:6.2f}%"
            )

        print()

        print("=" * 70)
        print(
            f" RETRIEVAL SCORE      : {score:3d}/100"
        )
        print("=" * 70)

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def _document_matches(
        self,
        document,
        expected_document: str,
    ) -> bool:
        """
        Compare the retrieved document with the expected one.

        The canonical identifier is the source filename.
        """

        if expected_document is None:
            return False

        return (
            document.source.name == expected_document
            or document.title == expected_document
        )

    # ---------------------------------------------------------

    def _guess_failure_reason(
        self,
        purpose: str,
        expected_rank: int | None,
        acceptable: bool,
    ) -> str:

        if acceptable:

            return (
                "Retrieved an acceptable section, "
                "but not the preferred one."
            )

        if expected_rank is not None:

            return (
                f"Expected section found at rank "
                f"{expected_rank}. "
                "Ranking needs improvement."
            )

        reasons = {

            "TITLE_SCORE":
                "Title matching failed.",

            "BODY_SCORE":
                "Body relevance too weak.",

            "PATH_SCORE":
                "Hierarchy/path information not sufficiently weighted.",

            "SECTION_HIERARCHY":
                "Parent/child hierarchy incorrectly ranked.",

            "INTENT_DESCRIPTION":
                "Description intent not recognised.",

            "INTENT_PROCEDURE":
                "Procedure intent not recognised.",

            "COMPLETE_MATCH":
                "Overall ranking favoured another section.",

            "RERANKING":
                "Semantic reranking required.",

            "THRESHOLD":
                "Confidence threshold missing. Retrieval should return no result.",

            "COVERAGE_SCORE":
                "Section coverage insufficient.",

            "MULTI_SECTION":
                "Answer likely spans multiple sections.",

            "SYNONYMS":
                "Synonym expansion required.",

            "PRONOUNS":
                "Contextual retrieval required.",
        }

        return reasons.get(
            purpose,
            "Unknown retrieval failure.",
        )

    # ---------------------------------------------------------

    def _load_dataset(
        self,
    ):

        with self._dataset_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    # ---------------------------------------------------------

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

            parser.parse(
                document,
            )

            for document in raw_documents
        ]

        return extractor.extract(
            documents,
        )

    # ---------------------------------------------------------

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





















































































