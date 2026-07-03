from __future__ import annotations

import time

from knowledge_engine.loaders.directory_loader import DirectoryLoader
from knowledge_engine.parsers.text_parser import TextParser
from knowledge_engine.semantic.knowledge_extractor import KnowledgeExtractor

from tests.benchmark.metrics.benchmark_metrics import BenchmarkMetrics


class SemanticRunner:
    """
    Benchmark del Knowledge Extractor.
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
        print(" SEMANTIC BENCHMARK")
        print("=" * 70)
        print()

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

        start = time.perf_counter()

        knowledge_base = extractor.extract(
            documents,
        )

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        print(
            f"Documents     : {len(knowledge_base.documents)}"
        )

        print(
            f"Entities      : {len(knowledge_base.entities)}"
        )

        print(
            f"Relations     : {len(knowledge_base.relations)}"
        )

        print(
            f"Extraction    : {elapsed:.2f} ms"
        )

        if knowledge_base.documents:

            print(
                f"Entities/Doc  : {len(knowledge_base.entities) / len(knowledge_base.documents):.2f}"
            )

            print(
                f"Relations/Doc : {len(knowledge_base.relations) / len(knowledge_base.documents):.2f}"
            )

        print()
        print("Semantic benchmark completed.")