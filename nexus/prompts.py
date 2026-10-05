"""
prompts.py
The grounded prompt NEXUS sends when local documents are in play.

Each retrieved chunk is numbered and labelled with its file, so the model can
cite `[1]`, `[2]` and the UI can line those citations up with the sources it
shows under the answer.
"""

from __future__ import annotations

SYSTEM_TEMPLATE = """You are NEXUS, a local assistant answering from the \
user's own documents. Use the numbered context below when it is relevant, and \
cite it inline with its number, e.g. [1] or [2][3]. If the context does not \
contain the answer, say so plainly, then answer from general knowledge \
without citing.

Context:
{context}
"""

NO_CONTEXT = "(No relevant local documents found.)"


def format_context(chunks: list[dict]) -> str:
    if not chunks:
        return NO_CONTEXT
    return "\n\n---\n\n".join(
        f"[{i}] {c['meta'].get('source', 'unknown')}\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    )


def build_prompt(question: str, chunks: list[dict]) -> str:
    system = SYSTEM_TEMPLATE.format(context=format_context(chunks))
    return f"{system}\nUser question: {question}\nAnswer:"
