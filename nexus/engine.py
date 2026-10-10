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

Cloud: when the cloud switch allows it, the chain can include Gemini, Groq,
OpenRouter, DeepSeek or Mistral models (see cloud.py). Every cloud call is
checked again against the privacy rules right before it is made, a failed
provider (rate limit, bad key, no internet) is skipped and remembered, and
the next model in the chain answers -- usually a local one.

`arena()` and `council()` ask more than one model the same question; both are
built on `answer()`, so every rule above applies to each model they ask.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from nexus import cloud, config, ollama, pc_agent, providers, store
from nexus.config import get_settings
from nexus.prompts import build_prompt
from nexus.retrieve import get_retriever
from nexus.router import TaskRouter

log = logging.getLogger(__name__)

RAG_MODES = ("auto", "always", "never")
ROLES = ("user", "assistant")

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
    # Cloud overrides; None means "use nexus.toml" (where cloud is off by default).
    cloud_mode: str | None = None    # off | hard | allowed
    allow_docs: bool | None = None   # let document-grounded questions leave the PC
    allow_paid: bool | None = None   # allow per-token-billed models

    def normalised(self) -> Options:
        mode = self.rag_mode if self.rag_mode in RAG_MODES else "auto"
        return Options(
            force_model=self.force_model or None,
            force_task=self.force_task or None,
            rag_mode=mode,
            top_k=max(1, min(int(self.top_k), 50)),
            rerank=bool(self.rerank),
            temperature=max(0.0, min(float(self.temperature), 2.0)),
            cloud_mode=self.cloud_mode or None,
            allow_docs=self.allow_docs,
            allow_paid=self.allow_paid,
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
# Conversation memory
# ---------------------------------------------------------------------------


def trim_history(
    history: list[dict[str, Any]] | None,
    max_messages: int | None = None,
    max_chars: int | None = None,
) -> list[dict[str, str]]:
    """The most recent turns within the memory budget in nexus.toml, oldest first.

    Only plain user/assistant text survives: UI metadata, empty turns and
    unknown roles are dropped. Whole messages are kept or dropped -- a turn
    cut in half reads as a different statement. What is kept here is then
    fitted to the answering model's context window by `fit_to_window`.
    """
    settings = get_settings()
    if max_messages is None:
        max_messages = settings.history_messages
    if max_chars is None:
        max_chars = settings.history_chars

    clean = [
        {"role": m["role"], "content": m["content"].strip()}
        for m in (history or [])
        if isinstance(m, dict)
        and m.get("role") in ROLES
        and isinstance(m.get("content"), str)
        and m["content"].strip()
    ]

    kept: list[dict[str, str]] = []
    used = 0
    for message in reversed(clean):
        if len(kept) >= max_messages:
            break
        size = len(message["content"])
        if used + size > max_chars:
            break
        kept.append(message)
        used += size
    kept.reverse()
    # A conversation sent to the model should open with the user speaking.
    while kept and kept[0]["role"] != "user":
        kept.pop(0)
    return kept


def without_document_turns(history: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """History minus answers that were grounded in your documents, and the
    questions that asked for them.

    This is what a cloud model sees when documents may not leave the PC: an
    earlier answer quoting your files is your files' content too.
    """
    items = [m for m in (history or []) if isinstance(m, dict)]
    drop: set[int] = set()
    for i, m in enumerate(items):
        meta = m.get("meta")
        if m.get("role") == "assistant" and isinstance(meta, dict) and (
            meta.get("needs_rag") or meta.get("sources")
        ):
            drop.add(i)
            if i > 0 and items[i - 1].get("role") == "user":
                drop.add(i - 1)
    return [m for i, m in enumerate(items) if i not in drop]


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
        "router_method": "rules",
        "p_strong": None,
        "history_used": 0,
        "local": True,        # False when a cloud model wrote the answer
        "cloud_mode": "off",
        "images": 0,
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
        router_method=decision.get("router_method", "rules"),
        p_strong=decision.get("p_strong"),
    )
    return result


# What both UIs save with an answer, so a reopened chat shows how it was made.
MESSAGE_META_KEYS = (
    "model", "provider", "local", "task", "auto_task", "router_method", "complexity",
    "needs_rag", "rag_reason", "sources", "chain", "attempts", "why", "info", "thinking",
    "elapsed", "prompt_chars", "history_used", "truncated", "chunks_dropped", "metrics",
    "cloud_mode", "images",
)


def message_meta(result: dict[str, Any]) -> dict[str, Any]:
    return {k: result.get(k) for k in MESSAGE_META_KEYS}


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


def _no_model_message(decision: dict[str, Any], needs_image: bool = False) -> str:
    if needs_image:
        return (
            "⚠️ No model that can read images is available. Pull a local vision "
            "model (`ollama pull qwen2.5vl:7b` or `ollama pull llava:7b`), or switch "
            "cloud on with a Gemini key set."
        )
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
        spec = providers.spec_by_name(model)
        existing = {"model": model, "provider": spec.provider if spec else "ollama",
                    "local": spec.is_local if spec else True, "score": 1.0,
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
# Generation
# ---------------------------------------------------------------------------

_FAILURE_WORDS = {
    cloud.RateLimited: "was rate-limited",
    cloud.AuthError: "rejected its API key",
    cloud.ModelNotFound: "doesn't serve this model any more",
    cloud.Offline: "was unreachable",
}


def _failure_note(provider: str, exc: Exception) -> str:
    label = cloud.PROVIDERS.get(provider, {}).get("label", provider)
    for kind, words in _FAILURE_WORDS.items():
        if isinstance(exc, kind):
            return f"{label} {words}"
    return f"{label} failed"


def _is_local(candidate: dict[str, Any]) -> bool:
    return bool(candidate.get("local", candidate.get("provider") in providers.LOCAL_PROVIDERS))


def _cloud_block(model: str, provider: str, task: str, needs_rag: bool,
                 policy: cloud.CloudPolicy) -> str | None:
    """Why this cloud model may not be called right now, or None if it may.

    Checked at call time, not only in planning: a pinned model, or a decision
    to use documents made after routing, must not slip past the rules.
    """
    if policy.mode == "off":
        return "cloud is switched off"
    spec = providers.spec_by_name(model)
    if spec is None:
        return "unknown cloud model"
    blocked = cloud.privacy_block(spec, task, needs_rag, policy)
    if blocked:
        return blocked
    status = cloud.provider_status(provider)
    return None if status["status"] == "ready" else status["detail"]


def _cloud_chat(
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    on_token: Callable[[str], None] | None = None,
    images: list[dict[str, str]] | None = None,
) -> ollama.ChatReply:
    """One cloud call, shaped like an Ollama reply so the caller treats both alike."""
    from nexus import cloud_client

    sent = time.monotonic()
    first: list[float] = []

    def timed(token: str) -> None:
        if not first:
            first.append(time.monotonic())
        if on_token:
            on_token(token)

    text = cloud_client.chat(provider, model, messages, temperature=temperature,
                             on_token=timed if on_token else None, images=images)
    ttft = round((first[0] - sent) * 1000, 1) if first else None
    return ollama.ChatReply(text, None, {"ttft_ms": ttft})


def _generate(provider: str, model: str, messages: list[dict[str, str]],
              temperature: float = 0.7) -> str:
    """A plain, unstreamed reply from one model (council judge, background work)."""
    if provider == "ollama":
        return ollama.chat(model, messages, temperature=temperature,
                           num_ctx=context_window(model)).text
    if provider in cloud.PROVIDERS:
        return _cloud_chat(provider, model, messages, temperature).text
    raise RuntimeError(f"No generator for provider {provider!r}")


def _record_overrides(question: str, opts: Options, decision: dict[str, Any]) -> None:
    """A task or document override is you correcting NEXUS: keep it as a
    training signal. Never allowed to break answering."""
    try:
        from nexus import feedback

        auto_task = decision.get("auto_task", decision.get("task"))
        if opts.force_task and opts.force_task != auto_task:
            feedback.record_signal("task_override", question, task=auto_task,
                                   value=opts.force_task)
        if opts.rag_mode in ("always", "never"):
            guess = getattr(get_router(), "_needs_rag", None)
            if guess is not None and bool(guess(question)) != (opts.rag_mode == "always"):
                feedback.record_signal("rag_override", question, task=decision.get("task"),
                                       value=opts.rag_mode)
    except Exception as exc:
        log.debug("could not record an override signal: %s", exc)


def _route_for(question: str, opts: Options, images: list) -> tuple[dict[str, Any], Any]:
    policy = cloud.CloudPolicy.from_settings(opts.cloud_mode, opts.allow_docs, opts.allow_paid)
    decision = get_router().route(
        question, force_task=opts.force_task or ("vision" if images else None), policy=policy,
        needs_image=bool(images), rag_override={"always": True, "never": False}.get(opts.rag_mode),
    )
    return decision, policy


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def answer(
    question: str,
    base_dir: Path | None = None,
    options: Options | None = None,
    on_token: Callable[[str], None] | None = None,
    on_thinking: Callable[[str], None] | None = None,
    history: list[dict[str, Any]] | None = None,
    chat_id: str | None = None,
    on_status: Callable[[str], None] | None = None,
    images: list[dict[str, str]] | None = None,
    record_signals: bool = True,
) -> dict[str, Any]:
    """Route `question`, then answer it with the best reachable model.

    `history` is the conversation so far ({"role", "content"}, oldest first);
    the newest turns within the memory budget are kept, then as much of them
    as fits the model's context window is sent. `images` are
    `{"data": <base64>, "mime": "image/png"}` dicts for this turn; with any
    attached, only image-reading models are considered. Returns the routing
    decision plus: answer, attempts, info, sources, thinking, truncated,
    chunks_dropped, metrics, requires_confirmation, pending, elapsed, local.

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
    images = [img for img in (images or []) if img.get("data")]
    decision, policy = _route_for(question, opts, images)
    result = _base_result(decision)
    result["cloud_mode"] = policy.mode
    result["images"] = len(images)
    if record_signals:
        _record_overrides(question, opts, decision)

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
    pinned = providers.spec_by_name(opts.force_model) if opts.force_model else None
    # A pinned cloud model is tried (and refused with a reason) even when
    # nothing else is reachable; a pinned local one needs Ollama to be up.
    if opts.force_model and (chain or (pinned and not pinned.is_local)):
        chain = _pin_model(chain, opts.force_model)
        result.update(chain=chain, model=chain[0]["model"], provider=chain[0]["provider"])

    if not chain:
        result["answer"] = _no_model_message(decision, needs_image=bool(images))
        result["info"] = "No model was reachable for this request."
        _finish(result, started, chat_id, error=result["info"])
        return result

    past = trim_history(history)
    result["history_used"] = len(past)
    # What a cloud model may see of the conversation when documents stay home.
    cloud_past = past if policy.allow_docs else trim_history(without_document_turns(history))

    status = on_status or (lambda _text: None)
    if opts.rag_mode != "never":
        status("Searching your documents…")
    chunks = _retrieve(result, question, past, opts)
    if (policy.mode != "off" and result["needs_rag"] and not policy.allow_docs
            and all(_is_local(c) for c in chain)):
        _note(result, "Cloud skipped: questions about your documents stay on this PC.")

    token_cb, thinking_cb = _guard(on_token), _guard(on_thinking)
    loaded = {m.lower() for m in decision.get("available", {}).get("loaded", [])}
    errors: list[str] = []
    for step, candidate in enumerate(chain):
        model, provider = candidate["model"], candidate["provider"]
        is_local = _is_local(candidate)

        if not is_local:
            blocked = _cloud_block(model, provider, result["task"], result["needs_rag"], policy)
            if blocked:
                result["attempts"].append({"model": model, "provider": provider,
                                           "error": f"skipped: {blocked}"})
                errors.append(f"{model}: {blocked}")
                if step == 0 and opts.force_model:
                    _note(result, f"Didn't use {model}: {blocked}.")
                continue
        elif provider != "ollama":
            error = f"no generator for provider {provider!r}"
            result["attempts"].append({"model": model, "provider": provider, "error": error})
            errors.append(f"{model}: {error}")
            continue

        try:
            if is_local:
                status(f"Writing with {model}…" if model.lower() in loaded
                       else f"Loading {model} — the first answer from a model takes longer…")
                num_ctx = context_window(model)
                fitted = fit_to_window(question, chunks, past, num_ctx, result["needs_rag"])
                think = "thinking" in providers.capabilities(model).get("caps", set())
                reply = ollama.chat(
                    model, fitted.messages, temperature=opts.temperature, num_ctx=num_ctx,
                    think=think, on_token=token_cb, on_thinking=thinking_cb, images=images,
                )
            else:
                label = cloud.PROVIDERS.get(provider, {}).get("label", provider)
                status(f"Asking {model} ({label}, cloud)…")
                spec = providers.spec_by_name(model)
                num_ctx = spec.context if spec else config.NUM_CTX
                fitted = fit_to_window(question, chunks, cloud_past, num_ctx, result["needs_rag"])
                reply = _cloud_chat(provider, model, fitted.messages, opts.temperature,
                                    on_token=token_cb, images=images)
        except _CallbackRaised as wrapped:
            raise wrapped.original from None
        except cloud.CloudError as exc:
            cloud.record_failure(exc, model)
            result["attempts"].append({"model": model, "provider": provider, "error": str(exc)})
            errors.append(f"{model}: {exc}")
            _note(result, f"{_failure_note(provider, exc)}.")
            continue
        except Exception as exc:
            log.info("%s failed, trying the next model: %s", model, exc)
            result["attempts"].append({"model": model, "provider": provider, "error": str(exc)})
            errors.append(f"{model}: {exc}")
            continue

        prompt_chars = sum(len(m["content"]) for m in fitted.messages)
        if not is_local:
            cloud.record_success(provider, prompt_chars, len(reply.text))
        result["attempts"].append({"model": model, "provider": provider, "error": None})
        result.update(
            model=model,
            provider=provider,
            local=is_local,
            why=candidate.get("reason"),
            answer=reply.text,
            thinking=reply.thinking,
            sources=[_source(c) for c in fitted.kept_chunks],
            chunks_dropped=fitted.chunks_dropped,
            prompt_chars=prompt_chars,
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


# ---------------------------------------------------------------------------
# More than one model: Arena and the council
# ---------------------------------------------------------------------------


def arena(
    question: str,
    options: Options | None = None,
    history: list[dict[str, Any]] | None = None,
    images: list[dict[str, str]] | None = None,
    chat_id: str | None = None,
    rng=None,
    on_status: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Answer `question` with two different models for a blind comparison.

    The pair is NEXUS's own first choice plus a challenger from the same
    routing chain -- so every privacy and cost rule that applies to normal
    answers applies here too. Answers are generated one after the other
    (an 8 GB card holds one local model at a time).

    Returns {"battle_id", "task", "a": {...}, "b": {...}, "note"}, or
    {"error": ...} when fewer than two models can answer.
    """
    from nexus import feedback

    opts = (options or Options()).normalised()
    images = [img for img in (images or []) if img.get("data")]
    decision, _policy = _route_for(question, opts, images)
    if decision["task"] == "system_agent":
        return {"error": "Arena compares model answers; PC actions don't use a model."}
    _record_overrides(question, opts, decision)
    pair = feedback.pick_pair(decision.get("chain") or [], rng)
    if pair is None:
        return {"error": "Arena needs at least two models that can answer this. "
                         "Pull another Ollama model or switch cloud on."}

    status = on_status or (lambda _text: None)
    sides = {}
    for label, candidate in zip(("a", "b"), pair):
        status(f"Answer {label.upper()} is being written…")
        result = answer(question, options=replace(opts, force_model=candidate["model"]),
                        history=history, images=images, chat_id=chat_id, record_signals=False)
        sides[label] = {
            "answer": result.get("answer", ""),
            "model": result.get("model"),
            "provider": result.get("provider"),
            "local": result.get("local", True),
            "intended": candidate["model"],
            "info": result.get("info"),
            "sources": result.get("sources") or [],
            "needs_rag": result.get("needs_rag"),
            "elapsed": result.get("elapsed"),
        }
    note = None
    if sides["a"]["model"] == sides["b"]["model"]:
        note = ("Both answers came from the same model (the other one was unavailable), "
                "so this vote won't change the leaderboard.")
    battle_id = feedback.create_battle(
        question, decision.get("task"), decision.get("complexity"),
        sides["a"], sides["b"], chat_id=chat_id,
    )
    return {"battle_id": battle_id, "task": decision.get("task"),
            "complexity": decision.get("complexity"), "a": sides["a"], "b": sides["b"],
            "note": note}


def council_recommended(question: str, options: Options | None = None) -> bool:
    """Should `[council] auto = "hard"` convene the council for this prompt?"""
    if get_settings().council_auto != "hard":
        return False
    opts = (options or Options()).normalised()
    decision, policy = _route_for(question, opts, [])
    if decision["task"] == "system_agent":
        return False
    hard = decision.get("p_strong")
    if hard is not None:
        return hard >= get_settings().strong_threshold
    return float(decision.get("complexity") or 0.0) >= policy.hard_threshold


def council(
    question: str,
    options: Options | None = None,
    history: list[dict[str, Any]] | None = None,
    images: list[dict[str, str]] | None = None,
    chat_id: str | None = None,
    on_status: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Ask several models, have a judge compare them, return one merged answer.

    Returns {"answer", "members": [...], "judge", "agreements",
    "disagreements", "agreement", "task", "judged", "note"} or {"error"}.
    """
    import concurrent.futures

    from nexus import council as council_mod

    opts = (options or Options()).normalised()
    images = [img for img in (images or []) if img.get("data")]
    decision, policy = _route_for(question, opts, images)
    if decision["task"] == "system_agent":
        return {"error": "The council compares model answers; PC actions don't use a model."}
    _record_overrides(question, opts, decision)
    settings = get_settings()
    members = council_mod.pick_members(decision.get("chain") or [], settings.council_members)
    if len(members) < 2:
        return {"error": "The council needs at least two models that can answer this. "
                         "Pull another Ollama model or switch cloud on."}
    status = on_status or (lambda _text: None)

    def ask(candidate: dict[str, Any]) -> dict[str, Any]:
        asked = time.monotonic()
        try:
            result = answer(question, options=replace(opts, force_model=candidate["model"]),
                            history=history, images=images, chat_id=chat_id,
                            record_signals=False)
        except Exception as exc:
            return {"model": candidate["model"], "provider": candidate["provider"],
                    "local": _is_local(candidate), "answer": "", "error": str(exc),
                    "elapsed": time.monotonic() - asked}
        return {"model": result.get("model"), "provider": result.get("provider"),
                "local": result.get("local", True), "answer": result.get("answer", ""),
                "error": None, "elapsed": result.get("elapsed"),
                "sources": result.get("sources") or [], "needs_rag": result.get("needs_rag"),
                "intended": candidate["model"]}

    # Cloud members in parallel while local ones run one at a time.
    cloud_members = [m for m in members if not _is_local(m)]
    local_members = [m for m in members if _is_local(m)]
    results: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(cloud_members))) as pool:
        futures = {pool.submit(ask, m): m["model"] for m in cloud_members}
        for m in local_members:
            status(f"Council: {m['model']} is answering…")
            results[m["model"]] = ask(m)
        for future, model in futures.items():
            results[model] = future.result()
    answered = [results[m["model"]] for m in members]
    # A member that fell back to a model already on the council adds nothing.
    seen, usable = set(), []
    for r in answered:
        if r["answer"] and r["model"] not in seen:
            seen.add(r["model"])
            usable.append(r)
    if not usable:
        return {"error": "No council member could answer: "
                         + "; ".join(f"{r['model']}: {r['error']}" for r in answered if r["error"])}

    texts = [r["answer"] for r in usable]
    agree = council_mod.agreement(texts)
    out = {"task": decision.get("task"), "complexity": decision.get("complexity"),
           "members": answered, "used": [r["model"] for r in usable],
           "agreement": agree["score"], "agreement_method": agree["method"],
           "agreements": [], "disagreements": [], "judge": None, "judged": False, "note": None,
           "sources": next((r.get("sources") for r in usable if r.get("sources")), []),
           "needs_rag": any(r.get("needs_rag") for r in usable)}
    if len(usable) == 1:
        out.update(answer=texts[0], note="Only one member answered, so there was nothing to compare.")
        return out

    # The judge sees every answer, so it must itself be allowed to see this
    # request: it comes from the same policy-filtered chain.
    chain = decision.get("chain") or []
    preferred = [c for c in chain if c["model"] == settings.council_judge] if settings.council_judge else []
    verdict, errors = None, []
    status("Council: the judge is comparing the answers…")
    for judge in preferred + [c for c in chain if c not in preferred]:
        if judge.get("provider") == "toolkit":
            continue
        if not _is_local(judge) and _cloud_block(judge["model"], judge["provider"], out["task"],
                                                 out["needs_rag"], policy):
            continue
        try:
            text = _generate(judge["provider"], judge["model"],
                             council_mod.judge_messages(question, texts), temperature=0.2)
        except Exception as exc:
            if isinstance(exc, cloud.CloudError):
                cloud.record_failure(exc, judge["model"])
            errors.append(f"{judge['model']}: {exc}")
            continue
        if not _is_local(judge):
            cloud.record_success(judge["provider"], sum(map(len, texts)), len(text))
        verdict = council_mod.parse_verdict(text, len(texts))
        if verdict:
            out["judge"] = judge["model"]
            break
        errors.append(f"{judge['model']}: reply was not usable JSON")
    if verdict:
        out.update(answer=verdict["final"], agreements=verdict["agreements"],
                   disagreements=verdict["disagreements"], judged=True)
    else:
        central = usable[agree["central"]]
        out.update(answer=central["answer"],
                   note=f"No judge could compare the answers ({'; '.join(errors[:2]) or 'none available'}); "
                        f"showing the answer most like the others, from {central['model']}.")
    return out


def council_message_meta(out: dict[str, Any], auto: bool = False) -> dict[str, Any]:
    """What a council answer stores with its chat message (both UIs use it)."""
    members = [{k: m.get(k) for k in ("model", "provider", "local", "answer", "error", "elapsed")}
               for m in out["members"]]
    judge_spec = providers.spec_by_name(out["judge"]) if out.get("judge") else None
    # "This PC" only if every member and the judge ran locally.
    local = (all(m.get("local", True) for m in members)
             and (judge_spec is None or judge_spec.is_local))
    info = {k: out.get(k) for k in ("judge", "judged", "agreement", "agreement_method",
                                    "agreements", "disagreements", "note", "used")}
    info.update(members=members, auto=auto)
    return {"model": "council", "provider": "council", "task": out.get("task"),
            "complexity": out.get("complexity"), "needs_rag": out.get("needs_rag"),
            "sources": out.get("sources") or [], "info": out.get("note"), "local": local,
            "council": info}


# ---------------------------------------------------------------------------
# Work NEXUS does on its own, and voice input
# ---------------------------------------------------------------------------


def local_generate(messages: list[dict[str, str]], task: str = "general",
                   temperature: float = 0.3) -> tuple[str, str]:
    """Generate with the best *local* model for `task`, whatever the cloud
    switch says. For work NEXUS does on its own (digests, flashcards): it
    reads your notes, and nothing leaves this PC unless you asked.

    Returns (text, model); raises RuntimeError when no local model answers.
    """
    avail = providers.availability(include_cloud=False)
    chain = providers.plan(task, "", 0.3, avail, needs_rag=True,
                           policy=cloud.CloudPolicy(mode="off"))
    errors = []
    for candidate in chain:
        if not candidate.spec.is_local:  # belt and braces: policy "off" has none
            continue
        try:
            return _generate("ollama", candidate.model, messages, temperature), candidate.model
        except Exception as exc:
            errors.append(f"{candidate.model}: {exc}")
    raise RuntimeError("No local model could answer"
                       + (": " + "; ".join(errors[:2]) if errors else " (is Ollama running?)"))


def transcribe(
    audio: bytes,
    filename: str = "recording.wav",
    mime: str = "audio/wav",
    options: Options | None = None,
) -> dict[str, Any]:
    """Voice input -> text, following the same cloud switch as answers."""
    from nexus import speech

    opts = (options or Options()).normalised()
    policy = cloud.CloudPolicy.from_settings(opts.cloud_mode, opts.allow_docs, opts.allow_paid)
    return speech.transcribe(audio, filename, mime, policy)
