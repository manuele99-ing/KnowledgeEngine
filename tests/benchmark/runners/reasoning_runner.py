from __future__ import annotations

import time

from knowledge_engine.llm.llm_client_factory import (
    create_llm_client,
)
from knowledge_engine.reasoning.reasoning_engine import (
    ReasoningEngine,
)

from tests.benchmark.metrics.benchmark_metrics import BenchmarkMetrics
from tests.benchmark.reports.benchmark_report import BenchmarkReport


class ReasoningRunner:
    """
    Executes the Reasoning benchmark.
    """

    def __init__(
        self,
        *,
        dataset: dict,
        knowledge_base,
        metrics: BenchmarkMetrics,
    ) -> None:

        self._dataset = dataset
        self._knowledge_base = knowledge_base
        self._metrics = metrics

    def run(self) -> None:

        print("=" * 70)
        print(" REASONING BENCHMARK")
        print("=" * 70)
        print()

        llm = create_llm_client()

        engine = ReasoningEngine(
            llm_client=llm,
        )

        for test in self._dataset["questions"]:

            expected = test["expected"]

            #
            # Skip negative questions
            #

            if expected.get(
                "expect_no_answer",
                False,
            ):
                continue

            self._metrics.total_questions += 1

            print(
                f"[{test['id']:03}] {test['question']}"
            )

            start = time.perf_counter()

            answer = engine.answer(
                question=test["question"],
                knowledge_base=self._knowledge_base,
            )

            elapsed = (
                time.perf_counter() - start
            ) * 1000

            self._metrics.add_answer_time(
                elapsed,
            )

            text = answer.text.lower()

            keywords = [
                keyword.lower()
                for keyword in expected.get(
                    "must_contain",
                    [],
                )
            ]

            found = sum(
                keyword in text
                for keyword in keywords
            )

            self._metrics.add_keyword_score(
                found=found,
                expected=len(keywords),
            )

            hallucinated = any(
                forbidden.lower() in text
                for forbidden in expected.get(
                    "must_not_contain",
                    [],
                )
            )

            self._metrics.add_hallucination(
                hallucinated,
            )

            success = (
                found
                >= len(keywords)
                * expected.get(
                    "minimum_keyword_score",
                    1.0,
                )
            )

            self._metrics.add_reasoning(
                success,
            )

            print(
                f"   Keywords : {found}/{len(keywords)}"
            )

            print(
                f"   Answer   : {elapsed:.2f} ms"
            )

            print()

        BenchmarkReport.print(
            self._metrics,
        )