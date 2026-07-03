from __future__ import annotations

import time

from knowledge_engine.loaders.directory_loader import DirectoryLoader
from knowledge_engine.parsers.text_parser import TextParser

from tests.benchmark.metrics.benchmark_metrics import BenchmarkMetrics


class ParserRunner:
    """
    Benchmark del parser.
    """

    def __init__(
        self,
        *,
        input_directory: str,
        metrics: BenchmarkMetrics,
    ) -> None:

        self._input_directory = input_directory
        self._metrics = metrics

    def run(self) -> None:

        print("=" * 70)
        print(" PARSER BENCHMARK")
        print("=" * 70)
        print()

        loader = DirectoryLoader()

        raw_documents = loader.load_directory(
            self._input_directory,
        )

        parser = TextParser()

        parsed_documents = 0
        total_sections = 0
        total_blocks = 0

        start = time.perf_counter()

        for raw_document in raw_documents:

            document = parser.parse(
                raw_document,
            )

            parsed_documents += 1

            total_sections += len(
                document.sections,
            )

            for section in document.sections:

                total_blocks += len(
                    section.blocks,
                )

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        print(f"Documents      : {parsed_documents}")
        print(f"Sections       : {total_sections}")
        print(f"Blocks         : {total_blocks}")
        print(f"Parsing Time   : {elapsed:.2f} ms")

        if parsed_documents:

            print(
                f"Avg Sections   : {total_sections / parsed_documents:.2f}"
            )

            print(
                f"Avg Blocks     : {total_blocks / parsed_documents:.2f}"
            )

        print()
        print("Parser benchmark completed.")