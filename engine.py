"""
engine.py
Single entry point that turns a user question into an answer:
route -> (system agent | best-fit model chain) -> text.

The router hands back an ordered chain of models chosen automatically from the
task, how hard the prompt looks, and what is actually reachable. This module
walks that chain top-down: the first model that answers wins, and every skip is
reported back to the UI in `info` and `attempts`. The user never picks a model.

Both the Streamlit UI (app.py) and the plain HTTP server (server.py) call
`answer()` so the routing/fallback behaviour stays in one place.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import requests

import pc_agent
from rag_pipeline import build_rag_prompt
from router import TaskRouter

PROJECT_DIR = Path(__file__).resolve().parent
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"


@lru_cache(maxsize=1)
def get_router() -> TaskRouter:
    return TaskRouter()


def _ollama_generate(model: str, prompt: str, timeout: int = 180) -> str:
    response = requests.post(
        OLLAMA_GENERATE_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    response.raise_for_status()
    text = response.json().get("response", "").strip()
    if not text:
        raise RuntimeError(f"{model} returned an empty response")
    return text


def _generate(provider: str, model: str, prompt: str) -> str:
    if provider == "ollama":
        return _ollama_generate(model, prompt)
    raise RuntimeError(f"No generator for provider {provider!r}")


def _base_result(decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": decision["task"],
        "model": decision.get("model"),
        "provider": decision.get("provider"),
        "chain": decision.get("chain", []),
        "complexity": decision.get("complexity", 0.0),
        "needs_rag": bool(decision.get("needs_rag")),
        "confidence": float(decision["confidence"]),
        "scores": {k: float(v) for k, v in decision["scores"].items()},
        "why": decision.get("reason"),
        "attempts": [],
        "info": None,
        "requires_confirmation": False,
        "pending": None,
    }


def _no_model_message(decision: dict[str, Any]) -> str:
    avail = decision.get("available", {})
    if not avail.get("ollama"):
        return (
            "⚠️ Ollama is not reachable. Start it with `ollama serve`, then pull "
            "a model with `ollama pull llama3.1:8b`. "
            "PC folder actions still work without it."
        )
    return (
        f"⚠️ None of your installed models can handle a **{decision['task']}** "
        "request. Pull a suitable Ollama model."
    )


def apply_pending(pending: dict[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
    """Execute a local action previously returned as `pending` by `answer()`."""
    outcome = pc_agent.apply(pending, base_dir or PROJECT_DIR)
    return {
        "answer": outcome["answer"],
        "task": "system_agent",
        "model": "pc-toolkit",
        "provider": "toolkit",
        "chain": [],
        "complexity": 0.0,
        "needs_rag": False,
        "confidence": 1.0,
        "scores": {},
        "why": None,
        "attempts": [],
        "info": None,
        "requires_confirmation": False,
        "pending": None,
    }


def answer(question: str, base_dir: Path | None = None, confirm: bool = False) -> dict[str, Any]:
    """Route `question`, then generate a reply with the best reachable model.

    Returns a dict: answer, task, model, provider, chain, complexity, needs_rag,
    confidence, scores, why, attempts, info, requires_confirmation, pending.

    `chain` is every model considered (best first) and `attempts` records what
    was actually tried, so the UI can show *why* a given AI answered. `info` is
    a non-fatal note (e.g. the first choice was down). When
    `requires_confirmation` is True, `pending` holds the local action to run via
    `apply_pending` once the user approves.
    """
    base_dir = base_dir or PROJECT_DIR
    decision = get_router().route(question)
    result = _base_result(decision)

    if decision["task"] == "system_agent":
        outcome = pc_agent.handle(question, base_dir, confirm=confirm)
        result["answer"] = outcome["answer"]
        result["requires_confirmation"] = outcome["requires_confirmation"]
        result["pending"] = outcome["pending"]
        result["model"] = "pc-toolkit"
        result["provider"] = "toolkit"
        # The word "folder" trips the RAG signal, but a file operation never
        # retrieves anything — don't claim it did.
        result["needs_rag"] = False
        return result

    chain = decision.get("chain") or []
    if not chain:
        result["answer"] = _no_model_message(decision)
        result["info"] = "No model was reachable for this request."
        return result

    # Ground the prompt in local documents once, then reuse it for whichever
    # model ends up answering.
    prompt = question
    if result["needs_rag"]:
        try:
            prompt = build_rag_prompt(question, top_k=5)
        except Exception as exc:
            result["info"] = f"Local document search unavailable ({exc}); answering without it."
            result["needs_rag"] = False

    errors: list[str] = []
    for step, candidate in enumerate(chain):
        model, provider = candidate["model"], candidate["provider"]
        try:
            text = _generate(provider, model, prompt)
        except Exception as exc:
            result["attempts"].append({"model": model, "provider": provider, "error": str(exc)})
            errors.append(f"{model}: {exc}")
            continue

        result["attempts"].append({"model": model, "provider": provider, "error": None})
        result["model"] = model
        result["provider"] = provider
        result["why"] = candidate.get("reason")
        if step > 0:
            first = chain[0]["model"]
            note = f"Auto-switched from {first} to {model} — first choice was unavailable."
            result["info"] = f"{result['info']} {note}".strip() if result["info"] else note
        result["answer"] = text
        return result

    raise RuntimeError(
        "Every available model failed for this request:\n  " + "\n  ".join(errors)
    )
