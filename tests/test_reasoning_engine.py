from __future__ import annotations

import unittest

from knowledge_engine.domain.document.document_source import (
    DocumentSource,
    SourceType,
)
from knowledge_engine.domain.document.raw_document import RawDocument
from knowledge_engine.llm.static_llm_client import StaticLLMClient
from knowledge_engine.parsers.text_parser import TextParser
from knowledge_engine.reasoning.reasoning_engine import ReasoningEngine
from knowledge_engine.semantic.knowledge_extractor import KnowledgeExtractor


class ReasoningEngineTest(unittest.TestCase):
    def test_answer_calls_llm_with_knowledge_base_context(self) -> None:
        knowledge_base = self._build_knowledge_base()
        llm_client = StaticLLMClient(
            response_text="Hub 2 uses the Jeweller protocol."
        )
        engine = ReasoningEngine(llm_client=llm_client)

        answer = engine.answer(
            question="Which protocol does Hub 2 use?",
            knowledge_base=knowledge_base,
        )

        self.assertEqual(answer.text, "Hub 2 uses the Jeweller protocol.")
        self.assertEqual(answer.question, "Which protocol does Hub 2 use?")
        self.assertEqual(len(llm_client.requests), 1)

        prompt = llm_client.requests[0].prompt

        self.assertIn("Knowledge base context:", prompt)
        self.assertIn("Ajax Hub 2", prompt)
        self.assertIn("Hub 2", prompt)
        self.assertIn("Jeweller", prompt)
        self.assertIn("The communication protocol is Jeweller.", prompt)
        self.assertIn("Which protocol does Hub 2 use?", prompt)
        self.assertIn("Hub 2", answer.context)

    def _build_knowledge_base(self):
        source = DocumentSource(
            type=SourceType.TEXT,
            name="manuale.txt",
            uri="test_data/manuale.txt",
        )
        raw_document = RawDocument(
            source=source,
            content=(
                "Ajax Hub 2\n\n"
                "The Hub 2 supports up to 100 devices.\n"
                "The communication protocol is Jeweller."
            ),
        )
        document = TextParser().parse(raw_document)

        return KnowledgeExtractor().extract(document)


if __name__ == "__main__":
    unittest.main()
