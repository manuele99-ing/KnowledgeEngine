from __future__ import annotations

import json
import time
from pathlib import Path

from knowledge_engine.loaders.directory_loader import DirectoryLoader
from knowledge_engine.parsers.text_parser import TextParser
from knowledge_engine.retrieval.retrieval_engine import RetrievalEngine
from knowledge_engine.semantic.knowledge_extractor import KnowledgeExtractor

from tests.benchmark.metrics.benchmark_metrics import BenchmarkMetrics
from tests.benchmark.reports.benchmark_report import BenchmarkReport


class BenchmarkRunner:
    """
    Executes a benchmark dataset against KnowledgeEngine.
    """

    def __init__(
        self,
        dataset_path: str |Path,
        input_directory: str = "input",
    ) -> None:

        self._dataset_path = Path(dataset_path)
        self._input_directory = input_directory

    def run(self) -> None:

        dataset = self._load_dataset()

        retrieval_engine = self._build_retrieval_engine()

        metrics = BenchmarkMetrics()

        print("=" * 70)
        print(" KNOWLEDGEENGINE BENCHMARK")
        print("=" * 70)
        print()

        print(f"Dataset : {dataset['name']}")
        print(f"Version : {dataset['version']}")
        print()

        for test in dataset["questions"]:

            metrics.total_questions += 1

            print(
                f"[{test['id']:03}] {test['question']}"
            )

            start = time.perf_counter()

            results = retrieval_engine.search(
                question=test["question"],
                top_k=1,
            )

            elapsed = (
                time.perf_counter() - start
            ) * 1000

            metrics.add_search_time(elapsed)

            if not results:

                print("   ✗ No result")
                print()

                continue

            result = results[0]

            expected = test["expected"]

            document_ok = (
                result.document.source.name
                == expected["document"]
            )

            section_ok = (
                result.section.title
                == expected["section"]
            )

            metrics.add_retrieval(
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

            found_keywords = sum(
                keyword in answer
                for keyword in keywords
            )

            metrics.add_keyword_score(
                found=found_keywords,
                expected=len(keywords),
            )

            hallucinated = any(
                forbidden.lower() in answer
                for forbidden in expected.get(
                    "must_not_contain",
                    [],
                )
            )

            metrics.add_hallucination(
                hallucinated,
            )

            reasoning_ok = (
                found_keywords
                >= len(keywords)
                * expected.get(
                    "minimum_keyword_score",
                    1.0,
                )
            )

            metrics.add_reasoning(
                reasoning_ok,
            )

            print(
                f"   Document : {'✓' if document_ok else '✗'}"
            )

            print(
                f"   Section  : {'✓' if section_ok else '✗'}"
            )

            print(
                f"   Keywords : {found_keywords}/{len(keywords)}"
            )

            print(
                f"   Search   : {elapsed:.2f} ms"
            )

            print()

        BenchmarkReport.print(metrics)

    def _load_dataset(
        self,
    ) -> dict:

        with self._dataset_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    def _build_retrieval_engine(
        self,
    ) -> RetrievalEngine:

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

        knowledge_base = extractor.extract(
            documents,
        )

        return RetrievalEngine(
            knowledge_base=knowledge_base,
        )


def main() -> None:

    BenchmarkRunner(
        dataset_path="tests/benchmark/datasets/disney_v1.json",
    ).run()


if __name__ == "__main__":
    main()