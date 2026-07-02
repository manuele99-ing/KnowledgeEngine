from __future__ import annotations


class PromptBuilder:
    """
    Builds the prompt sent to the language model.
    """

    def build(
        self,
        question: str,
        context: str,
    ) -> str:

        context_section = (
            f"Contesto:\n\n{context}"
            if context
            else "Contesto:\n\nNessun contesto disponibile."
        )

        return f"""
Sei un assistente che risponde alle domande utilizzando esclusivamente la documentazione fornita.

Regole:
- Leggi solo il contesto fornito.
- Non usare informazioni esterne.
- Rispondi con una sola frase chiara in italiano.
- Se il contesto contiene una risposta alla domanda, usa quella informazione.
- Non ripetere il contesto parola per parola.
- Se la risposta non è presente nel contesto, rispondi esattamente:
"Non lo so."

{context_section}

Domanda:
{question}

Risposta:
""".strip()
