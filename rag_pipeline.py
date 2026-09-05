"""
rag_pipeline.py
Ties retrieval to generation: pulls relevant chunks, builds a grounded
prompt, and sends it to a locally running Ollama model.

Prereq: `ollama serve` running, and a model already pulled, e.g.:
    ollama pull llama3.1:8b

Usage:
    python rag_pipeline.py
"""

from functools import lru_cache

import requests

from retrieve import Retriever

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.1:8b"  # swap for whichever model your router picks

# Chunks are token-sized (<=240) rather than 1600 characters, so each one
# carries roughly a quarter of what it used to. eval_rag.py measured the
# consequence directly: at top_k=5 the model received 1096 context tokens
# and grounded@5 fell to 0.944; at top_k=10 it gets 2190 tokens and
# grounded@5 returns to 1.000. Retrieval depth has to follow chunk size.
DEFAULT_TOP_K = 10

SYSTEM_TEMPLATE = """You are NEXUS AI's local assistant. Answer the user's \
question using ONLY the context below when it's relevant. If the context \
doesn't contain the answer, say so plainly and answer from general \
knowledge instead. Cite the source filename in brackets when you use it.

Context:
{context}
"""


def build_prompt(question: str, chunks: list) -> str:
    if not chunks:
        context = "(No relevant local documents found.)"
    else:
        context_parts = []
        for c in chunks:
            source = c["meta"].get("source", "unknown")
            context_parts.append(f"[{source}]\n{c['text']}")
        context = "\n\n---\n\n".join(context_parts)

    system = SYSTEM_TEMPLATE.format(context=context)
    return f"{system}\n\nUser question: {question}\nAnswer:"


def ask_ollama(prompt: str, model: str = DEFAULT_MODEL) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def check_ollama_model(model: str = DEFAULT_MODEL) -> None:
    """Fail early when Ollama or the selected model is unavailable."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        installed_models = {item.get("name") for item in response.json().get("models", [])}
    except requests.RequestException as exc:
        raise RuntimeError(
            "Ollama is not reachable. Start it with `ollama serve` and run this command again."
        ) from exc

    if model not in installed_models:
        available = ", ".join(sorted(name for name in installed_models if name)) or "none"
        raise RuntimeError(
            f"Ollama model `{model}` is not installed. Available models: {available}. "
            f"Install it with `ollama pull {model}` or change DEFAULT_MODEL."
        )


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()


def retrieve_context(
    question: str, top_k: int = DEFAULT_TOP_K, rerank: bool = True
) -> tuple[str, list[dict]]:
    """Retrieve local context and return `(grounded_prompt, chunks)`.

    The chunks come back alongside the prompt so a caller can show *what*
    grounded an answer -- which file, which score, and which retrieval arm
    found it -- instead of the user having to trust that retrieval worked.
    """
    chunks = get_retriever().query(question, top_k=top_k, rerank=rerank)
    return build_prompt(question, chunks), chunks


def build_rag_prompt(question: str, top_k: int = DEFAULT_TOP_K) -> str:
    """Retrieve local context and return a grounded prompt."""
    return retrieve_context(question, top_k=top_k)[0]


def rag_query(question: str, model: str = DEFAULT_MODEL, top_k: int = DEFAULT_TOP_K) -> str:
    return ask_ollama(build_rag_prompt(question, top_k), model=model)


if __name__ == "__main__":
    try:
        check_ollama_model()
    except RuntimeError as exc:
        raise SystemExit(f"NEXUS setup error: {exc}") from exc

    print("NEXUS AI RAG test (Ctrl+C to quit)")
    while True:
        try:
            q = input("\nYou: ")
        except KeyboardInterrupt:
            break
        answer = rag_query(q)
        print(f"\nNEXUS: {answer}")
