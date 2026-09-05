"""
engine.py
Single entry point that turns a user question into an answer:
route -> (system agent | best-fit model chain) -> text.

The router hands back an ordered chain of models chosen automatically from the
task, how hard the prompt looks, and what is actually reachable. This module
walks that chain top-down: the first model that answers wins, and every skip is
reported back to the UI in `info` and `attempts`.

Automatic selection stays the default. `Options` exists so a caller can
*override* it deliberately -- pin a model, force retrieval on or off, change
retrieval depth -- which is what makes the routing inspectable rather than
something the user has to take on trust.

Both the Streamlit UI (app.py) and the plain HTTP server (server.py) call
`answer()` so the routing/fallback behaviour stays in one place.
"""

import json
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import requests

import pc_agent
import providers
from rag_pipeline import DEFAULT_TOP_K, retrieve_context
from router import TaskRouter

PROJECT_DIR = Path(__file__).resolve().parent
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"

RAG_MODES = ("auto", "always", "never")


@dataclass
class Options:
    """Deliberate overrides of the automatic behaviour.

    Defaults reproduce the fully automatic path exactly, so `Options()` is
    always safe to pass.
    """

    force_model: str | None = None   # pin a specific model
    force_task: str | None = None    # override the classifier
    rag_mode: str = "auto"           # auto | always | never
    top_k: int = DEFAULT_TOP_K
    rerank: bool = True
    temperature: float = 0.7

    def normalised(self) -> "Options":
        mode = self.rag_mode if self.rag_mode in RAG_MODES else "auto"
        return Options(
            force_model=self.force_model or None,
            force_task=self.force_task or None,
            rag_mode=mode,
            top_k=max(1, min(int(self.top_k), 50)),
            rerank=bool(self.rerank),
            temperature=max(0.0, min(float(self.temperature), 2.0)),
        )


@lru_cache(maxsize=1)
def get_router() -> TaskRouter:
    return TaskRouter()


def _ollama_generate(
    model: str,
    prompt: str,
    temperature: float = 0.7,
    timeout: int = 300,
    on_token: Callable[[str], None] | None = None,
) -> str:
    """Generate with Ollama, streaming token-by-token when `on_token` is given."""
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": on_token is not None,
        "options": {"temperature": float(temperature)},
    }

    if on_token is None:
        response = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=timeout)
        response.raise_for_status()
        text = response.json().get("response", "")
    else:
        parts: list[str] = []
        with requests.post(
            OLLAMA_GENERATE_URL, json=payload, timeout=timeout, stream=True
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = chunk.get("response", "")
                if token:
                    parts.append(token)
                    on_token(token)
                if chunk.get("done"):
                    break
        text = "".join(parts)

    text = text.strip()
    if not text:
        raise RuntimeError(f"{model} returned an empty response")
    return text


def _generate(
    provider: str,
    model: str,
    prompt: str,
    temperature: float = 0.7,
    on_token: Callable[[str], None] | None = None,
) -> str:
    if provider == "ollama":
        return _ollama_generate(model, prompt, temperature=temperature, on_token=on_token)
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
        "auto_task": decision.get("auto_task", decision["task"]),
        "attempts": [],
        "info": None,
        "requires_confirmation": False,
        "pending": None,
        "sources": [],
        "elapsed": 0.0,
        "prompt_chars": 0,
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


def _pin_model(chain: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    """Move `model` to the front of the chain, adding it if it isn't there.

    The rest of the chain is kept as fallback so a pinned model that fails to
    load degrades instead of dead-ending; `info` records that it happened.
    """
    rest = [c for c in chain if c["model"] != model]
    existing = next((c for c in chain if c["model"] == model), None)
    if existing is None:
        spec = providers.BY_NAME.get(model)
        existing = {
            "model": model,
            "provider": spec.provider if spec else "ollama",
            "score": 1.0,
            "reason": f"{model}: pinned by you",
        }
    else:
        existing = dict(existing)
        existing["reason"] = f"{existing.get('reason', model)} (pinned by you)"
    return [existing] + rest


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
        "sources": [],
        "elapsed": 0.0,
        "prompt_chars": 0,
    }


def answer(
    question: str,
    base_dir: Path | None = None,
    confirm: bool = False,
    options: Options | None = None,
    on_token: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Route `question`, then generate a reply with the best reachable model.

    Returns a dict: answer, task, model, provider, chain, complexity, needs_rag,
    confidence, scores, why, attempts, info, requires_confirmation, pending,
    sources, elapsed, prompt_chars.

    `chain` is every model considered (best first) and `attempts` records what
    was actually tried, so the UI can show *why* a given AI answered. `sources`
    holds the retrieved chunks behind a grounded answer. Pass `on_token` to
    stream the reply as it is produced.
    """
    started = time.monotonic()
    base_dir = base_dir or PROJECT_DIR
    opts = (options or Options()).normalised()

    decision = get_router().route(question, force_task=opts.force_task)
    result = _base_result(decision)

    if decision["task"] == "system_agent":
        outcome = pc_agent.handle(question, base_dir, confirm=confirm)
        result["answer"] = outcome["answer"]
        result["requires_confirmation"] = outcome["requires_confirmation"]
        result["pending"] = outcome["pending"]
        result["model"] = "pc-toolkit"
        result["provider"] = "toolkit"
        # The word "folder" trips the RAG signal, but a file operation never
        # retrieves anything -- don't claim it did.
        result["needs_rag"] = False
        result["elapsed"] = time.monotonic() - started
        return result

    chain = decision.get("chain") or []
    if opts.force_model and chain:
        chain = _pin_model(chain, opts.force_model)
        result["chain"] = chain
        result["model"] = chain[0]["model"]
        result["provider"] = chain[0]["provider"]

    if not chain:
        result["answer"] = _no_model_message(decision)
        result["info"] = "No model was reachable for this request."
        result["elapsed"] = time.monotonic() - started
        return result

    # Retrieval gate: the router's keyword guess by default, or whatever the
    # caller asked for.
    if opts.rag_mode == "always":
        result["needs_rag"] = True
    elif opts.rag_mode == "never":
        result["needs_rag"] = False

    # Ground the prompt in local documents once, then reuse it for whichever
    # model ends up answering.
    prompt = question
    if result["needs_rag"]:
        try:
            prompt, chunks = retrieve_context(
                question, top_k=opts.top_k, rerank=opts.rerank
            )
            result["sources"] = [
                {
                    "source": c["meta"].get("source", "unknown"),
                    "chunk_index": c["meta"].get("chunk_index"),
                    "score": c.get("score"),
                    "rerank_score": c.get("rerank_score"),
                    "vector_rank": c.get("vector_rank"),
                    "bm25_rank": c.get("bm25_rank"),
                    "text": c["text"],
                }
                for c in chunks
            ]
            if not chunks:
                result["info"] = "No local documents matched; answering without them."
        except Exception as exc:
            result["info"] = f"Local document search unavailable ({exc}); answering without it."
            result["needs_rag"] = False
    result["prompt_chars"] = len(prompt)

    errors: list[str] = []
    for step, candidate in enumerate(chain):
        model, provider = candidate["model"], candidate["provider"]
        try:
            text = _generate(
                provider, model, prompt, temperature=opts.temperature, on_token=on_token
            )
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
        result["elapsed"] = time.monotonic() - started
        return result

    raise RuntimeError(
        "Every available model failed for this request:\n  " + "\n  ".join(errors)
    )
