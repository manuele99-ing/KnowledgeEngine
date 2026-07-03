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
        print("=" * 72)
        print(" KNOWLEDGEENGINE BENCHMARK REPORT")
        print("=" * 72)
        print()

        print(f"Questions               : {metrics.total_questions}")
        print()

        print("RETRIEVAL")
        print("-" * 72)
        print(
            f"Document Accuracy       : {metrics.document_score:6.2f}%"
        )
        print(
            f"Section Accuracy        : {metrics.section_score:6.2f}%"
        )
        print()

        print("REASONING")
        print("-" * 72)
        print(
            f"Reasoning Score         : {metrics.reasoning_score:6.2f}%"
        )
        print()

        print("ANSWER QUALITY")
        print("-" * 72)
        print(
            f"Keyword Coverage        : {metrics.keyword_score:6.2f}%"
        )
        print(
            f"Hallucination Score     : {metrics.hallucination_score:6.2f}%"
        )
        print(
            f"Negative Questions      : {metrics.negative_score:6.2f}%"
        )
        print()

        print("PERFORMANCE")
        print("-" * 72)
        print(
            f"Average Search Time     : {metrics.average_search_time_ms:6.2f} ms"
        )
        print(
            f"Average Answer Time     : {metrics.average_answer_time_ms:6.2f} ms"
        )
        print()

        print("=" * 72)
        print(
            f"OVERALL SCORE           : {metrics.overall_score:6.2f}/100"
        )
        print("=" * 72)