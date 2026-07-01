from __future__ import annotations

from knowledge_engine.domain.knowledge.knowledge_base import KnowledgeBase
from knowledge_engine.reasoning.reasoning_engine import ReasoningEngine


class ChatSession:
    """
    Simple terminal chat session.
    """

    def __init__(
        self,
        reasoning_engine: ReasoningEngine,
        knowledge_base: KnowledgeBase,
    ) -> None:
        self._reasoning_engine = reasoning_engine
        self._knowledge_base = knowledge_base

    def run(self) -> None:
        print("\n" + "=" * 60)
        print("Interactive Chat")
        print("Type 'exit' to quit.")
        print("=" * 60)

        while True:
            question = input("\n> ").strip()

            if not question:
                continue

            if question.lower() == "exit":
                print("\nGoodbye!")
                break

            answer = self._reasoning_engine.answer(
                question=question,
                knowledge_base=self._knowledge_base,
            )

            print(f"\n{answer.text}")

