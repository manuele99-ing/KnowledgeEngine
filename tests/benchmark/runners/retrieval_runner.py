from __future__ import annotations

import time

from knowledge_engine.retrieval.retrieval_engine import RetrievalEngine

from tests.benchmark.metrics.benchmark_metrics import BenchmarkMetrics
from tests.benchmark.reports.benchmark_report import BenchmarkReport


class RetrievalRunner:
    """
    Executes the Retrieval benchmark.
    """

    def __init__(
        self,
        *,
        dataset: dict,
        retrieval_engine: RetrievalEngine,
        metrics: BenchmarkMetrics,
    ) ->None:

        self._dataset = dataset
        self._retrieval_engine = retrieval_engine
        self._metrics = metrics

    def run(self) -> None:

        questions = self._dataset["questions"]

        print("=" * 70)
        print(" RETRIEVAL BENCHMARK")
        print("=" * 70)
        print()

        for test in questions:

            self._metrics.total_questions += 1

            print(
                f"[{test['id']:03}] {test['question']}"
            )

            start = time.perf_counter()

            results = self._retrieval_engine.search(
                question=test["question"],
                top_k=self._dataset["settings"].get(
                    "top_k",
                    1,
                ),
            )

            elapsed = (
                time.perf_counter() - start
            ) * 1000.0

            self._metrics.add_search_time(
                elapsed,
            )

            expected = test["expected"]

            #
            # Negative question
            #

            if expected.get(
                "expect_no_answer",
                False,
            ):

                success = len(results) == 0

                self._metrics.add_negative(
                    success,
                )

                print(
                    f"   Negative : {'✓' if success else '✗'}"
                )

                print(
                    f"   Search   : {elapsed:.2f} ms"
                )

                print()

                continue

            #
            # No result
            #

            if not results:

                print("   ✗ No result")
                print()

                continue

            result = results[0]

            document_ok = (
                result.document.source.name
                == expected["document"]
            )

            section_ok = (
                result.section.title
                == expected["section"]
            )

            self._metrics.add_retrieval(
                document_ok=document_ok,
                section_ok=section_ok,
            )

            answer = "\n".join(
                block.content
                for block in result.section.blocks
            ).lower()

            keywords = [
                keyword.lower()
                for keyword in expected.get(
                    "must_contain",
                    [],
                )
            ]

            found = sum(
                keyword in answer
                for keyword in keywords
            )

            self._metrics.add_keyword_score(
                found=found,
                expected=len(keywords),
            )

            hallucinated = any(
                forbidden.lower() in answer
                for forbidden in expected.get(
                    "must_not_contain",
                    [],
                )
            )

            self._metrics.add_hallucination(
                hallucinated,
            )

            reasoning_ok = (
                found
                >= len(keywords)
                * expected.get(
                    "minimum_keyword_score",
                    1.0,
                )
            )

            self._metrics.add_reasoning(
                reasoning_ok,
            )

            print(
                f"   Document : {'✓' if document_ok else '✗'}"
            )

            print(
                f"   Section  : {'✓' if section_ok else '✗'}"
            )

            print(
                f"   Keywords : {found}/{len(keywords)}"
            )

            print(
                f"   Search   : {elapsed:.2f} ms"
            )

            print()

        BenchmarkReport.print(
            self._metrics,
        )