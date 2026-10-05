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

Both UIs (app.py, server.py) and the CLI call `answer()`, so the routing and
fallback behaviour lives in one place.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from nexus import config, ollama, pc_agent, providers, store
from nexus.prompts import build_prompt
from nexus.retrieve import get_retriever
from nexus.router import TaskRouter

log = logging.getLogger(__name__)

RAG_MODES = ("auto", "always", "never")

REPLY_RESERVE = 1024     # tokens kept free for the model's answer
TRUNCATION_RATIO = 0.98  # prompt_eval_count this close to num_ctx => probably cut off
FOLLOW_UP_WORDS = 12     # a message shorter than this, with history, is a follow-up
# Cross-encoder logit above which the best retrieved chunk counts as relevant
# even though the question named no document. Measured on the golden set:
# 15 of 18 document questions score above it, 0 of 15 off-topic ones do
# (their best is -4.2, typically -8 to -11).
RELEVANCE_THRESHOLD = -3.0


@dataclass
class Options:
    """Deliberate overrides of the automatic behaviour.

    Defaults reproduce the fully automatic path exactly, so `Options()` is
    always safe to pass.
    """

    force_model: str | None = None   # pin a specific model
    force_task: str | None = None    # override the classifier
    rag_mode: str = "auto"           # auto | always | never
    top_k: int = config.DEFAULT_TOP_K
    rerank: bool = True
    temperature: float = 0.7

    def normalised(self) -> Options:
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


class _CallbackRaised(Exception):
    """An exception from the caller's on_token/on_thinking callback, carried
    out of the fallback chain so it is never mistaken for the model failing."""

    def __init__(self, original: Exception):
        super().__init__(repr(original))
        self.original = original


def _guard(callback: Callable[[str], None] | None) -> Callable[[str], None] | None:
    if callback is None:
        return None

    def call(text: str) -> None:
        try:
            callback(text)
        except Exception as exc:
            raise _CallbackRaised(exc) from exc

    return call


# ---------------------------------------------------------------------------
# Prompt budgeting
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Deliberately pessimistic: ~3 chars per token, where English on Llama/Qwen
    tokenizers runs ~4. Over-estimating means sending slightly less — never
    having Ollama silently cut the front off."""
    return math.ceil(len(text) / 3)


def context_window(model: str) -> int:
    reported = providers.capabilities(model).get("context_length")
    return min(config.NUM_CTX, reported) if reported else config.NUM_CTX


@dataclass
class Fitted:
    messages: list[dict[str, str]]
    kept_chunks: list[dict[str, Any]]
    chunks_dropped: int
    over_budget: bool


def fit_to_window(
    question: str,
    chunks: list[dict[str, Any]],
    history: list[dict[str, str]],
    num_ctx: int,
    use_rag: bool,
) -> Fitted:
    """Build one model's chat messages so they fit its context window.

    Priority: the question (with the RAG template) always; then retrieved
    chunks in rank order; then history, newest first, whole messages only.
    """
    budget = num_ctx - REPLY_RESERVE

    def render(kept: list[dict[str, Any]]) -> str:
        return build_prompt(question, kept) if use_rag else question

    used = estimate_tokens(render([]))
    over = used > budget  # sent anyway: refusing would be worse than a cut prompt

    kept: list[dict[str, Any]] = []
    if use_rag and not over:
        for candidate in chunks:
            cost = estimate_tokens(render(kept + [candidate]))
            if cost > budget:
                break
            kept.append(candidate)
            used = cost

    past: list[dict[str, str]] = []
    if not over:
        for msg in reversed(history):
            cost = estimate_tokens(msg["content"]) + 4  # role/framing overhead
            if used + cost > budget:
                break
            past.insert(0, {"role": msg["role"], "content": msg["content"]})
            used += cost

    return Fitted(
        past + [{"role": "user", "content": render(kept)}],
        kept,
        len(chunks) - len(kept),
        over,
    )


def _retrieval_query(question: str, history: list[dict[str, str]]) -> str:
    """"and the second one?" retrieves nothing useful alone — borrow the
    previous question. Only the search query changes, never the prompt."""
    if history and len(question.split()) < FOLLOW_UP_WORDS:
        previous = next((m["content"] for m in reversed(history) if m["role"] == "user"), None)
        if previous:
            return f"{previous} {question}"
    return question


# ---------------------------------------------------------------------------
# Result shaping
# ---------------------------------------------------------------------------


def _source(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": chunk["meta"].get("source", "unknown"),
        "chunk_index": chunk["meta"].get("chunk_index"),
        "score": chunk.get("score"),
        "rerank_score": chunk.get("rerank_score"),
        "vector_rank": chunk.get("vector_rank"),
        "bm25_rank": chunk.get("bm25_rank"),
        "text": chunk["text"],
    }


def _note(result: dict[str, Any], text: str) -> None:
    result["info"] = f"{result['info']} {text}" if result.get("info") else text


def _empty_result(task: str) -> dict[str, Any]:
    """Every key a caller may read, so no UI has to guess which exist."""
    return {
        "answer": "",
        "task": task,
        "model": None,
        "provider": None,
        "chain": [],
        "complexity": 0.0,
        "needs_rag": False,
        "rag_reason": None,
        "confidence": 0.0,
        "scores": {},
        "why": None,
        "auto_task": task,
        "attempts": [],
        "info": None,
        "requires_confirmation": False,
        "pending": None,
        "sources": [],
        "elapsed": 0.0,
        "prompt_chars": 0,
        "thinking": None,
        "truncated": False,
        "chunks_dropped": 0,
        "stopped": False,
        "metrics": {
            "ttft_ms": None, "total_ms": None, "load_ms": None,
            "prompt_tokens": None, "eval_tokens": None, "tokens_per_s": None,
        },
    }


def _base_result(decision: dict[str, Any]) -> dict[str, Any]:
    result = _empty_result(decision["task"])
    result.update(
        model=decision.get("model"),
        provider=decision.get("provider"),
        chain=decision.get("chain", []),
        complexity=decision.get("complexity", 0.0),
        needs_rag=bool(decision.get("needs_rag")),
        rag_reason="your question mentions your documents" if decision.get("needs_rag") else None,
        confidence=float(decision["confidence"]),
        scores={k: float(v) for k, v in decision["scores"].items()},
        why=decision.get("reason"),
        auto_task=decision.get("auto_task", decision["task"]),
    )
    return result


def _finish(
    result: dict[str, Any], started: float, chat_id: str | None, error: str | None = None
) -> None:
    """Stamp timings and record exactly one `turns` row. Never raises."""
    result["elapsed"] = time.monotonic() - started
    metrics = result["metrics"]
    metrics["total_ms"] = round(result["elapsed"] * 1000, 1)
    row = {
        "chat_id": chat_id,
        "task": result.get("task"),
        "model": None if error else result.get("model"),
        "attempts": result.get("attempts"),
        "rag_used": bool(result.get("sources")),
        "chunks_dropped": result.get("chunks_dropped", 0),
        "truncated": result.get("truncated", False),
        "stopped": False,
        "error": error,
        **{k: metrics.get(k) for k in
           ("ttft_ms", "total_ms", "load_ms", "prompt_tokens", "eval_tokens", "tokens_per_s")},
    }
    try:
        store.record_turn(row)
    except Exception as exc:
        log.warning("could not record metrics: %s", exc)


def _no_model_message(decision: dict[str, Any]) -> str:
    if not decision.get("available", {}).get("ollama"):
        return (
            "⚠️ Ollama is not reachable. Start it with `ollama serve`, then pull "
            f"a model with `ollama pull {config.DEFAULT_PULL_MODEL}`. "
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
        existing = {"model": model, "provider": "ollama", "score": 1.0,
                    "reason": f"{model}: pinned by you"}
    else:
        existing = dict(existing, reason=f"{existing.get('reason', model)} (pinned by you)")
    return [existing] + rest


def apply_pending(pending: dict[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
    """Execute a local action previously returned as `pending` by `answer()`."""
    outcome = pc_agent.apply(pending, base_dir or config.PROJECT_DIR)
    result = _empty_result("system_agent")
    result.update(answer=outcome["answer"], model="pc-toolkit", provider="toolkit", confidence=1.0)
    return result


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def _retrieve(
    result: dict[str, Any], question: str, history: list[dict[str, str]], opts: Options
) -> list[dict[str, Any]]:
    """Decide whether to ground this answer in local documents, and fetch them.

    `always`/`never` are absolute. In `auto`, a question that names the
    documents ("what does the readme say") always retrieves; any other
    question still gets a quick look, and is grounded only if the best chunk
    clears RELEVANCE_THRESHOLD -- so "which model writes the answer?" finds
    the notes that say so instead of being answered from general knowledge.
    """
    if opts.rag_mode == "never":
        result["needs_rag"], result["rag_reason"] = False, None
        return []
    if opts.rag_mode == "always":
        result["needs_rag"], result["rag_reason"] = True, "you asked to always use documents"
    probing = not result["needs_rag"]
    if probing and not opts.rerank:
        return []  # the threshold is calibrated on cross-encoder scores

    try:
        chunks = get_retriever().query(
            _retrieval_query(question, history), top_k=opts.top_k, rerank=opts.rerank
        )
    except Exception as exc:
        log.warning("document search failed: %s", exc)
        if not probing:
            _note(result, f"Local document search unavailable ({exc}); answering without it.")
        result["needs_rag"] = False
        return []

    if probing:
        best = chunks[0].get("rerank_score") if chunks else None
        if best is None or best < RELEVANCE_THRESHOLD:
            return []
        result["needs_rag"] = True
        result["rag_reason"] = f"your documents looked relevant (best match {best:+.1f})"
    elif not chunks:
        _note(result, "No local documents matched; answering without them.")
    return chunks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def answer(
    question: str,
    base_dir: Path | None = None,
    options: Options | None = None,
    on_token: Callable[[str], None] | None = None,
    on_thinking: Callable[[str], None] | None = None,
    history: list[dict[str, str]] | None = None,
    chat_id: str | None = None,
    on_status: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Route `question`, then answer it with the best reachable model.

    `history` is the conversation so far ({"role", "content"}, oldest first);
    as much as fits the model's context window is sent, newest first. Returns
    the routing decision plus: answer, attempts, info, sources, thinking,
    truncated, chunks_dropped, metrics, requires_confirmation, pending, elapsed.

    `chain` is every model considered (best first) and `attempts` records what
    was actually tried, so the UI can show *why* a given AI answered. One
    `turns` row is recorded per call. `on_status` receives short progress
    lines ("Searching your documents…") for the wait before the first token.
    Exceptions raised by any callback propagate unchanged and are never
    treated as a model failure.
    """
    started = time.monotonic()
    base_dir = base_dir or config.PROJECT_DIR
    opts = (options or Options()).normalised()
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in (history or [])
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]

    decision = get_router().route(question, force_task=opts.force_task)
    result = _base_result(decision)

    if decision["task"] == "system_agent":
        outcome = pc_agent.handle(question, base_dir)
        result.update(
            answer=outcome["answer"],
            requires_confirmation=outcome["requires_confirmation"],
            pending=outcome["pending"],
            model="pc-toolkit",
            provider="toolkit",
            # The word "folder" trips the RAG signal, but a file operation
            # never retrieves anything -- don't claim it did.
            needs_rag=False,
            rag_reason=None,
        )
        _finish(result, started, chat_id)
        return result

    chain = decision.get("chain") or []
    if opts.force_model and chain:
        chain = _pin_model(chain, opts.force_model)
        result.update(chain=chain, model=chain[0]["model"], provider=chain[0]["provider"])

    if not chain:
        result["answer"] = _no_model_message(decision)
        result["info"] = "No model was reachable for this request."
        _finish(result, started, chat_id, error=result["info"])
        return result

    status = on_status or (lambda _text: None)
    if opts.rag_mode != "never":
        status("Searching your documents…")
    chunks = _retrieve(result, question, history, opts)

    token_cb, thinking_cb = _guard(on_token), _guard(on_thinking)
    loaded = {m.lower() for m in decision.get("available", {}).get("loaded", [])}
    errors: list[str] = []
    for step, candidate in enumerate(chain):
        model, provider = candidate["model"], candidate["provider"]
        if provider != "ollama":
            error = f"no generator for provider {provider!r}"
            result["attempts"].append({"model": model, "provider": provider, "error": error})
            errors.append(f"{model}: {error}")
            continue

        status(f"Writing with {model}…" if model.lower() in loaded
               else f"Loading {model} — the first answer from a model takes longer…")
        num_ctx = context_window(model)
        fitted = fit_to_window(question, chunks, history, num_ctx, result["needs_rag"])
        think = "thinking" in providers.capabilities(model).get("caps", set())
        try:
            reply = ollama.chat(
                model, fitted.messages, temperature=opts.temperature, num_ctx=num_ctx,
                think=think, on_token=token_cb, on_thinking=thinking_cb,
            )
        except _CallbackRaised as wrapped:
            raise wrapped.original from None
        except Exception as exc:
            log.info("%s failed, trying the next model: %s", model, exc)
            result["attempts"].append({"model": model, "provider": provider, "error": str(exc)})
            errors.append(f"{model}: {exc}")
            continue

        result["attempts"].append({"model": model, "provider": provider, "error": None})
        result.update(
            model=model,
            provider=provider,
            why=candidate.get("reason"),
            answer=reply.text,
            thinking=reply.thinking,
            sources=[_source(c) for c in fitted.kept_chunks],
            chunks_dropped=fitted.chunks_dropped,
            prompt_chars=sum(len(m["content"]) for m in fitted.messages),
        )
        result["metrics"].update(reply.metrics)
        if fitted.chunks_dropped:
            _note(result, f"{fitted.chunks_dropped} of {len(chunks)} retrieved chunks were "
                          f"left out to fit {model}'s {num_ctx}-token context window.")
        prompt_tokens = reply.metrics.get("prompt_tokens") or 0
        if fitted.over_budget or prompt_tokens >= TRUNCATION_RATIO * num_ctx:
            result["truncated"] = True
            _note(result, f"The prompt filled {model}'s {num_ctx}-token context window, "
                          "so the start of it may have been cut off.")
        if step > 0:
            _note(result, f"Auto-switched from {chain[0]['model']} to {model} — "
                          "first choice was unavailable.")
        _finish(result, started, chat_id)
        return result

    message = "Every available model failed for this request:\n  " + "\n  ".join(errors)
    _finish(result, started, chat_id, error=message)
    raise RuntimeError(message)
