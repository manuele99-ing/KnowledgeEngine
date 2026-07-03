from __future__ import annotations

import argparse
import json
from pathlib import Path

from knowledge_engine.loaders.directory_loader import DirectoryLoader
from knowledge_engine.parsers.text_parser import TextParser
from knowledge_engine.retrieval.retrieval_engine import RetrievalEngine
from knowledge_engine.semantic.knowledge_extractor import KnowledgeExtractor

from tests.benchmark.metrics.benchmark_metrics import BenchmarkMetrics


class BenchmarkRunner:

    def __init__(
        self,
        dataset_path: str,
        input_directory: str = "input",
    ) -> None:

        self._dataset_path = Path(dataset_path)
        self._input_directory = input_directory

    def run(
        self,
        module: str,
    ) -> None:

        dataset = self._load_dataset()

        if module == "parser":

            from tests.benchmark.runners.parser_runner import (
                ParserRunner,
            )

            ParserRunner(
                input_directory=self._input_directory,
                metrics=BenchmarkMetrics(),
            ).run()

            return

        if module == "semantic":

            from tests.benchmark.runners.semantic_runner import (
                SemanticRunner,
            )

            SemanticRunner(
                input_directory=self._input_directory,
                metrics=BenchmarkMetrics(),
            ).run()

            return

        if module == "retrieval":

            from tests.benchmark.runners.retrieval_runner import (
                RetrievalRunner,
            )

            RetrievalRunner(
                dataset=dataset,
                retrieval_engine=self._build_retrieval_engine(),
                metrics=BenchmarkMetrics(),
            ).run()

            return

        if module == "reasoning":

            from tests.benchmark.runners.reasoning_runner import (
                ReasoningRunner,
            )

            ReasoningRunner(
                dataset=dataset,
                knowledge_base=self._build_knowledge_base(),
                metrics=BenchmarkMetrics(),
            ).run()

            return

        if module == "full":

            from tests.benchmark.runners.parser_runner import (
                ParserRunner,
            )

            from tests.benchmark.runners.semantic_runner import (
                SemanticRunner,
            )

            from tests.benchmark.runners.retrieval_runner import (
                RetrievalRunner,
            )

            from tests.benchmark.runners.reasoning_runner import (
                ReasoningRunner,
            )

            print("\n==============================")
            print("PARSER")
            print("==============================\n")

            ParserRunner(
                input_directory=self._input_directory,
                metrics=BenchmarkMetrics(),
            ).run()

            print("\n==============================")
            print("SEMANTIC")
            print("==============================\n")

            SemanticRunner(
                input_directory=self._input_directory,
                metrics=BenchmarkMetrics(),
            ).run()

            print("\n==============================")
            print("RETRIEVAL")
            print("==============================\n")

            RetrievalRunner(
                dataset=dataset,
                retrieval_engine=self._build_retrieval_engine(),
                metrics=BenchmarkMetrics(),
            ).run()

            print("\n==============================")
            print("REASONING")
            print("==============================\n")

            ReasoningRunner(
                dataset=dataset,
                knowledge_base=self._build_knowledge_base(),
                metrics=BenchmarkMetrics(),
            ).run()

            return

        raise ValueError(
            f"Unknown benchmark module: {module}"
        )

    def _load_dataset(
        self,
    ) -> dict:

        with self._dataset_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    def _build_knowledge_base(self):

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
    ) -> RetrievalEngine:

        return RetrievalEngine(
            knowledge_base=self._build_knowledge_base(),
        )


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--module",
        default="retrieval",
        choices=[
            "parser",
            "semantic",
            "retrieval",
            "reasoning",
            "full",
        ],
    )

    parser.add_argument(
        "--dataset",
        default="tests/benchmark/datasets/disney_v2.json",
    )

    parser.add_argument(
        "--input",
        default="input",
    )

    args = parser.parse_args()

    BenchmarkRunner(
        dataset_path=args.dataset,
        input_directory=args.input,
    ).run(
        module=args.module,
    )


if __name__ == "__main__":
    main()