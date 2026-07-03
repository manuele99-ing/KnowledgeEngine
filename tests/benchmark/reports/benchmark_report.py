from __future__ import annotations

from tests.benchmark.metrics.benchmark_metrics import BenchmarkMetrics


class BenchmarkReport:
    """
    Prints the final benchmark report.
    """

    @staticmethod
    def print(
        metrics: BenchmarkMetrics,
    ) -> None:

        print()
        print("=" * 70)
        print(" KNOWLEDGEENGINE QUALITY REPORT")
        print("=" * 70)
        print()

        print(f"Questions                : {metrics.total_questions}")
        print()

        print("Retrieval")
        print("-" * 70)
        print(
            f"Document Accuracy        : {metrics.document_score:6.2f}%"
        )
        print(
            f"Section Accuracy         : {metrics.section_score:6.2f}%"
        )
        print()

        print("Reasoning")
        print("-" * 70)
        print(
            f"Reasoning Score          : {metrics.reasoning_score:6.2f}%"
        )
        print()

        print("Answer Quality")
        print("-" * 70)
        print(
            f"Keyword Coverage         : {metrics.keyword_score:6.2f}%"
        )
        print(
            f"Hallucination Score      : {metrics.hallucination_score:6.2f}%"
        )
        print()

        print("Performance")
        print("-" * 70)
        print(
            f"Average Search Time      : {metrics.average_search_time_ms:6.2f} ms"
        )
        print(
            f"Average Answer Time      : {metrics.average_answer_time_ms:6.2f} ms"
        )
        print()

        print("=" * 70)
        print(
            f"OVERALL QUALITY SCORE    : {metrics.overall_score:6.2f}/100"
        )
        print("=" * 70)