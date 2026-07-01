from __future__ import annotations

from knowledge_engine.retrieval.retrieved_section import RetrievedSection


class PromptBuilder:
    """
    Builds the prompt sent to the language model.
    """

    def build(
        self,
        question: str,
        sections: list[RetrievedSection],
    ) -> str:

        context_parts: list[str] = []

        for item in sections:

            text = "\n".join(
                block.content
                for block in item.section.blocks
            )

            context_parts.append(
                f"""
Documento: {item.document.source}
Sezione: {item.section.title}

{text}
""".strip()
            )

        context = "\n\n------------------------------\n\n".join(
            context_parts
        )

        return f"""
Sei un assistente che risponde alle domande utilizzando esclusivamente la documentazione fornita.

Regole:

- Usa solo le informazioni presenti nel contesto.
- Non inventare informazioni.
- Se la risposta non è presente nella documentazione, rispondi esattamente:
"Non lo so."

Contesto:

{context}

Domanda:

{question}

Risposta:
""".strip()