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
`answer()` so the routing/fallback behaviour stays in one place. Streamlit also
passes the conversation so far as `history`.
"""

import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import requests

import pc_agent
import providers
import store
from rag_pipeline import DEFAULT_TOP_K, build_prompt, get_retriever
from router import TaskRouter

PROJECT_DIR = Path(__file__).resolve().parent

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


OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
# 5 s to connect, then at most 120 s of silence between streamed chunks. A long
# answer is fine while tokens keep arriving; a stuck model fails and the chain
# moves on. Loading a 7B model (~30 s here) happens before the first chunk.
OLLAMA_TIMEOUT = (5, 120)
# Ollama's default window is 2k-4k tokens depending on version, and it silently
# drops the front of anything longer. Always ask for this much explicitly.
NUM_CTX = int(os.environ.get("NEXUS_NUM_CTX") or 8192)

_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.S)


@dataclass
class ChatReply:
    text: str
    thinking: str | None
    metrics: dict[str, Any]


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


def _split_inline_thinking(text: str, thinking: str | None) -> tuple[str, str | None]:
    """Older Ollama versions put reasoning inline as <think>…</think>."""
    parts = [thinking] if thinking else []
    match = _THINK_BLOCK.search(text)
    if match:
        parts.append(match.group(1).strip())
        text = _THINK_BLOCK.sub("", text, count=1)
    elif "</think>" in text:
        # Some chat templates open the block in the prompt, so only the close arrives.
        head, _, text = text.partition("</think>")
        parts.append(head.strip())
    joined = "\n".join(p for p in parts if p)
    return text.strip(), joined or None


def _metrics(final: dict[str, Any], sent: float, first: float | None) -> dict[str, Any]:
    eval_count = final.get("eval_count")
    eval_seconds = (final.get("eval_duration") or 0) / 1e9
    load_ns = final.get("load_duration")
    return {
        "ttft_ms": round((first - sent) * 1000, 1) if first is not None else None,
        "load_ms": round(load_ns / 1e6, 1) if load_ns is not None else None,
        "prompt_tokens": final.get("prompt_eval_count"),
        "eval_tokens": eval_count,
        "tokens_per_s": (
            round(eval_count / eval_seconds, 2) if eval_count and eval_seconds > 0 else None
        ),
    }


def _ollama_chat(
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.7,
    num_ctx: int = NUM_CTX,
    think: bool = False,
    on_token: Callable[[str], None] | None = None,
    on_thinking: Callable[[str], None] | None = None,
) -> ChatReply:
    """One streamed /api/chat call.

    Raises on HTTP errors, error chunks and empty replies. Callback exceptions
    propagate unchanged; the `with` block closes the socket on the way out,
    which is what makes Ollama stop generating.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": float(temperature), "num_ctx": int(num_ctx)},
    }
    if think:
        payload["think"] = True

    sent = time.monotonic()
    first: float | None = None
    parts: list[str] = []
    thoughts: list[str] = []
    final: dict[str, Any] = {}
    with requests.post(
        OLLAMA_CHAT_URL, json=payload, timeout=OLLAMA_TIMEOUT, stream=True
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            if chunk.get("error"):
                raise RuntimeError(f"{model}: {chunk['error']}")
            message = chunk.get("message") or {}
            thought = message.get("thinking") or ""
            token = message.get("content") or ""
            if (thought or token) and first is None:
                first = time.monotonic()
            if thought:
                thoughts.append(thought)
                if on_thinking:
                    on_thinking(thought)
            if token:
                parts.append(token)
                if on_token:
                    on_token(token)
            if chunk.get("done"):
                final = chunk
                break

    text, thinking = _split_inline_thinking("".join(parts), "".join(thoughts) or None)
    if not text:
        raise RuntimeError(f"{model} returned an empty response")
    return ChatReply(text, thinking, _metrics(final, sent, first))


REPLY_RESERVE = 1024     # tokens kept free for the model's answer
TRUNCATION_RATIO = 0.98  # prompt_eval_count this close to num_ctx => probably cut off
FOLLOW_UP_WORDS = 12     # a message shorter than this, with history, is a follow-up


def estimate_tokens(text: str) -> int:
    """Deliberately pessimistic: ~3 chars per token, where English on Llama/Qwen
    tokenizers runs ~4. Over-estimating means sending slightly less — never
    having Ollama silently cut the front off."""
    return math.ceil(len(text) / 3)


def context_window(model: str) -> int:
    reported = providers.capabilities(model).get("context_length")
    return min(NUM_CTX, reported) if reported else NUM_CTX


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
        print(f"[nexus] could not record metrics: {exc}", file=sys.stderr)


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
        "thinking": None,
        "truncated": False,
        "chunks_dropped": 0,
        "stopped": False,
        "metrics": {
            "ttft_ms": None, "total_ms": None, "load_ms": None,
            "prompt_tokens": None, "eval_tokens": None, "tokens_per_s": None,
        },
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
    options: Options | None = None,
    on_token: Callable[[str], None] | None = None,
    on_thinking: Callable[[str], None] | None = None,
    history: list[dict[str, str]] | None = None,
    chat_id: str | None = None,
) -> dict[str, Any]:
    """Route `question`, then answer it with the best reachable model.

    `history` is the conversation so far ({"role", "content"}, oldest first);
    as much as fits the model's context window is sent, newest first. Returns
    the routing decision plus: answer, attempts, info, sources, thinking,
    truncated, chunks_dropped, metrics, requires_confirmation, pending, elapsed.

    `chain` is every model considered (best first) and `attempts` records what
    was actually tried, so the UI can show *why* a given AI answered. One
    `turns` row is recorded per call. Exceptions raised by `on_token` /
    `on_thinking` propagate unchanged and are never treated as a model failure.
    """
    started = time.monotonic()
    base_dir = base_dir or PROJECT_DIR
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
        result["answer"] = outcome["answer"]
        result["requires_confirmation"] = outcome["requires_confirmation"]
        result["pending"] = outcome["pending"]
        result["model"] = "pc-toolkit"
        result["provider"] = "toolkit"
        # The word "folder" trips the RAG signal, but a file operation never
        # retrieves anything -- don't claim it did.
        result["needs_rag"] = False
        _finish(result, started, chat_id)
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
        _finish(result, started, chat_id, error=result["info"])
        return result

    # Retrieval gate: the router's keyword guess by default, or the override.
    if opts.rag_mode == "always":
        result["needs_rag"] = True
    elif opts.rag_mode == "never":
        result["needs_rag"] = False

    chunks: list[dict[str, Any]] = []
    if result["needs_rag"]:
        try:
            chunks = get_retriever().query(
                _retrieval_query(question, history), top_k=opts.top_k, rerank=opts.rerank
            )
            if not chunks:
                _note(result, "No local documents matched; answering without them.")
        except Exception as exc:
            _note(result, f"Local document search unavailable ({exc}); answering without it.")
            result["needs_rag"] = False

    token_cb, thinking_cb = _guard(on_token), _guard(on_thinking)
    errors: list[str] = []
    for step, candidate in enumerate(chain):
        model, provider = candidate["model"], candidate["provider"]
        if provider != "ollama":
            error = f"no generator for provider {provider!r}"
            result["attempts"].append({"model": model, "provider": provider, "error": error})
            errors.append(f"{model}: {error}")
            continue

        num_ctx = context_window(model)
        fitted = fit_to_window(question, chunks, history, num_ctx, result["needs_rag"])
        think = "thinking" in providers.capabilities(model).get("caps", set())
        try:
            reply = _ollama_chat(
                model, fitted.messages, temperature=opts.temperature, num_ctx=num_ctx,
                think=think, on_token=token_cb, on_thinking=thinking_cb,
            )
        except _CallbackRaised as wrapped:
            raise wrapped.original
        except Exception as exc:
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
