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

Cloud: when the cloud switch allows it, the chain can include Gemini, Groq,
OpenRouter, DeepSeek or Mistral models (see cloud.py). Every cloud
call is checked again against the privacy rules right before it is made, a
failed provider (rate limit, bad key, no internet) is skipped and remembered,
and the next model in the chain answers -- usually a local one.

Conversation memory: `answer()` takes the earlier turns as `history` and sends
them through Ollama's chat endpoint, trimmed to the budget in `config.py`, so
"explain that more simply" knows what "that" is. Retrieved context goes in a
system message for the current turn only; it is never stored in history, so
old context can't pile up and crowd out the conversation.
"""

import json
import time
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import requests

import cloud
import pc_agent
import providers
from config import get_settings
from rag_pipeline import DEFAULT_TOP_K, build_system, retrieve_chunks
from router import TaskRouter

PROJECT_DIR = Path(__file__).resolve().parent
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

ROLES = ("user", "assistant")

# A follow-up this short ("and the second one?") carries too little to
# retrieve on by itself, so retrieval also sees the previous question.
FOLLOW_UP_WORDS = 8

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
    # Cloud overrides; None means "use nexus.toml" (where cloud is off by default).
    cloud_mode: str | None = None    # off | hard | allowed
    allow_docs: bool | None = None   # let document-grounded questions leave the PC
    allow_paid: bool | None = None   # allow per-token-billed models

    def normalised(self) -> "Options":
        mode = self.rag_mode if self.rag_mode in RAG_MODES else "auto"
        return Options(
            force_model=self.force_model or None,
            force_task=self.force_task or None,
            rag_mode=mode,
            top_k=max(1, min(int(self.top_k), 50)),
            rerank=bool(self.rerank),
            cloud_mode=self.cloud_mode or None,
            allow_docs=self.allow_docs,
            allow_paid=self.allow_paid,
            temperature=max(0.0, min(float(self.temperature), 2.0)),
        )


@lru_cache(maxsize=1)
def get_router() -> TaskRouter:
    return TaskRouter()


def trim_history(
    history: list[dict[str, Any]] | None,
    max_messages: int | None = None,
    max_chars: int | None = None,
) -> list[dict[str, str]]:
    """The most recent turns that fit the memory budget, oldest first.

    Only plain user/assistant text survives: UI metadata, empty turns and
    unknown roles are dropped. Whole messages are kept or dropped -- a turn
    cut in half reads as a different statement.
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
    items = list(history or [])
    drop: set[int] = set()
    for i, m in enumerate(items):
        meta = m.get("meta") if isinstance(m, dict) else None
        if m.get("role") == "assistant" and isinstance(meta, dict) and (
            meta.get("needs_rag") or meta.get("sources")
        ):
            drop.add(i)
            if i > 0 and items[i - 1].get("role") == "user":
                drop.add(i - 1)
    return [m for i, m in enumerate(items) if i not in drop]


def build_messages(
    question: str,
    history: list[dict[str, str]] | None = None,
    system: str | None = None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend(history or [])
    messages.append({"role": "user", "content": question})
    return messages


def retrieval_query(question: str, history: list[dict[str, str]] | None) -> str:
    """What to search the documents with: the question, plus the previous
    question when this one is a short follow-up."""
    if not history or len(question.split()) >= FOLLOW_UP_WORDS:
        return question
    previous = next((m["content"] for m in reversed(history) if m["role"] == "user"), None)
    return f"{previous}\n{question}" if previous else question


def _with_ollama_images(
    messages: list[dict[str, str]], images: list[dict[str, str]] | None
) -> list[dict[str, Any]]:
    """Ollama takes images as a base64 list on the user message they belong to."""
    out: list[dict[str, Any]] = [dict(m) for m in messages]
    if images:
        for i in range(len(out) - 1, -1, -1):
            if out[i]["role"] == "user":
                out[i]["images"] = [img["data"] for img in images]
                break
    return out


def _ollama_chat(
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    timeout: int = 300,
    on_token: Callable[[str], None] | None = None,
    images: list[dict[str, str]] | None = None,
) -> str:
    """Chat with Ollama, streaming token-by-token when `on_token` is given."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": _with_ollama_images(messages, images),
        "stream": on_token is not None,
        "options": {"temperature": float(temperature)},
    }

    if on_token is None:
        response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=timeout)
        response.raise_for_status()
        text = (response.json().get("message") or {}).get("content", "")
    else:
        parts: list[str] = []
        with requests.post(
            OLLAMA_CHAT_URL, json=payload, timeout=timeout, stream=True
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = (chunk.get("message") or {}).get("content", "")
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
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    on_token: Callable[[str], None] | None = None,
    images: list[dict[str, str]] | None = None,
) -> str:
    if provider == "ollama":
        return _ollama_chat(model, messages, temperature=temperature, on_token=on_token,
                            images=images)
    if provider in cloud.PROVIDERS:
        import cloud_client

        return cloud_client.chat(provider, model, messages, temperature=temperature,
                                 on_token=on_token, images=images)
    raise RuntimeError(f"No generator for provider {provider!r}")


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
        "router_method": decision.get("router_method", "rules"),
        "p_strong": decision.get("p_strong"),
        "attempts": [],
        "info": None,
        "requires_confirmation": False,
        "pending": None,
        "sources": [],
        "elapsed": 0.0,
        "prompt_chars": 0,
        "history_used": 0,
        "local": True,
        "cloud_mode": "off",
        "images": 0,
    }


def _no_model_message(decision: dict[str, Any], needs_image: bool = False) -> str:
    avail = decision.get("available", {})
    if needs_image:
        return (
            "⚠️ No model that can read images is available. Pull a local vision "
            "model (`ollama pull qwen2.5vl:7b` or `ollama pull llava:7b`), or switch "
            "cloud on with a Gemini key set."
        )
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
        spec = providers.spec_by_name(model)
        existing = {
            "model": model,
            "provider": spec.provider if spec else "ollama",
            "local": spec.is_local if spec else True,
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
    history: list[dict[str, Any]] | None = None,
    images: list[dict[str, str]] | None = None,
    record_signals: bool = True,
) -> dict[str, Any]:
    """Route `question`, then generate a reply with the best reachable model.

    `history` is the conversation so far (oldest first, `role`/`content`
    dicts, not including `question`). It is trimmed to the memory budget.
    `images` are `{"data": <base64>, "mime": "image/png"}` dicts for this
    turn; with any attached, only image-reading models are considered.

    Returns a dict: answer, task, model, provider, chain, complexity, needs_rag,
    confidence, scores, why, attempts, info, requires_confirmation, pending,
    sources, elapsed, prompt_chars, history_used.

    `chain` is every model considered (best first) and `attempts` records what
    was actually tried, so the UI can show *why* a given AI answered. `sources`
    holds the retrieved chunks behind a grounded answer. Pass `on_token` to
    stream the reply as it is produced.
    """
    started = time.monotonic()
    base_dir = base_dir or PROJECT_DIR
    opts = (options or Options()).normalised()
    images = [img for img in (images or []) if img.get("data")]
    policy = cloud.CloudPolicy.from_settings(opts.cloud_mode, opts.allow_docs, opts.allow_paid)

    # Whether documents are used is settled before routing, because it
    # decides whether this request may leave the PC at all.
    rag_override = {"always": True, "never": False}.get(opts.rag_mode)
    force_task = opts.force_task or ("vision" if images else None)
    decision = get_router().route(
        question, force_task=force_task, policy=policy,
        needs_image=bool(images), rag_override=rag_override,
    )
    result = _base_result(decision)
    result["cloud_mode"] = policy.mode
    result["images"] = len(images)
    if record_signals:
        _record_overrides(question, opts, decision)

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
    pinned_spec = providers.spec_by_name(opts.force_model) if opts.force_model else None
    if opts.force_model and (chain or (pinned_spec and not pinned_spec.is_local)):
        chain = _pin_model(chain, opts.force_model)
        result["chain"] = chain
        result["model"] = chain[0]["model"]
        result["provider"] = chain[0]["provider"]

    if not chain:
        result["answer"] = _no_model_message(decision, needs_image=bool(images))
        result["info"] = "No model was reachable for this request."
        result["elapsed"] = time.monotonic() - started
        return result

    notes: list[str] = []
    if (policy.mode != "off" and result["needs_rag"] and not policy.allow_docs
            and all(c.get("local", True) for c in chain)):
        notes.append("Cloud skipped: questions about your documents stay on this PC.")

    past = trim_history(history)
    result["history_used"] = len(past)

    # Ground the turn in local documents once, then reuse it for whichever
    # model ends up answering.
    system = None
    if result["needs_rag"]:
        try:
            chunks = retrieve_chunks(
                retrieval_query(question, past), top_k=opts.top_k, rerank=opts.rerank
            )
            system = build_system(chunks)
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
    if result["info"]:
        notes.insert(0, result["info"])
    messages = build_messages(question, past, system)
    result["prompt_chars"] = sum(len(m["content"]) for m in messages)
    cloud_messages = (
        messages if policy.allow_docs
        else build_messages(question, trim_history(without_document_turns(history)), system)
    )

    errors: list[str] = []
    for step, candidate in enumerate(chain):
        model, provider = candidate["model"], candidate["provider"]
        is_local = candidate.get("local", provider == "ollama")

        if not is_local:
            # Checked again here, not only in planning: a pinned model or a
            # retrieval decision made after routing must not slip past the rules.
            spec = providers.spec_by_name(model)
            if policy.mode == "off":
                blocked = "cloud is switched off"
            elif spec is None:
                blocked = "unknown cloud model"
            else:
                blocked = cloud.privacy_block(spec, result["task"], result["needs_rag"], policy)
            if not blocked:
                status = cloud.provider_status(provider)
                if status["status"] != "ready":
                    blocked = status["detail"]
            if blocked:
                result["attempts"].append({"model": model, "provider": provider,
                                           "error": f"skipped: {blocked}"})
                errors.append(f"{model}: {blocked}")
                if step == 0 and opts.force_model:
                    notes.append(f"Didn't use {model}: {blocked}.")
                continue

        try:
            text = _generate(
                provider, model, messages if is_local else cloud_messages,
                temperature=opts.temperature, on_token=on_token, images=images,
            )
        except cloud.CloudError as exc:
            cloud.record_failure(exc, model)
            result["attempts"].append({"model": model, "provider": provider, "error": str(exc)})
            errors.append(f"{model}: {exc}")
            notes.append(f"{_failure_note(provider, exc)}.")
            continue
        except Exception as exc:
            result["attempts"].append({"model": model, "provider": provider, "error": str(exc)})
            errors.append(f"{model}: {exc}")
            continue

        if not is_local:
            cloud.record_success(provider, result["prompt_chars"], len(text))
        result["attempts"].append({"model": model, "provider": provider, "error": None})
        result["model"] = model
        result["provider"] = provider
        result["local"] = bool(is_local)
        result["why"] = candidate.get("reason")
        if step > 0:
            first = chain[0]["model"]
            notes.append(f"Auto-switched from {first} to {model} — first choice was unavailable.")
        result["info"] = " ".join(notes) or None
        result["answer"] = text
        result["elapsed"] = time.monotonic() - started
        return result

    raise RuntimeError(
        "Every available model failed for this request:\n  " + "\n  ".join(errors)
    )


def _record_overrides(question: str, opts: Options, decision: dict[str, Any]) -> None:
    """A task or document override is you correcting NEXUS: keep it as a
    training signal. Never allowed to break answering."""
    try:
        import feedback

        auto_task = decision.get("auto_task", decision.get("task"))
        if opts.force_task and opts.force_task != auto_task:
            feedback.record_signal("task_override", question, task=auto_task,
                                   value=opts.force_task)
        if opts.rag_mode in ("always", "never"):
            guess = getattr(get_router(), "_needs_rag", None)
            if guess is not None and bool(guess(question)) != (opts.rag_mode == "always"):
                feedback.record_signal("rag_override", question, task=decision.get("task"),
                                       value=opts.rag_mode)
    except Exception:
        pass


def arena(
    question: str,
    options: Options | None = None,
    history: list[dict[str, Any]] | None = None,
    images: list[dict[str, str]] | None = None,
    chat_id: int | None = None,
    rng=None,
) -> dict[str, Any]:
    """Answer `question` with two different models for a blind comparison.

    The pair is NEXUS's own first choice plus a challenger from the same
    routing chain -- so every privacy and cost rule that applies to normal
    answers applies here too. Answers are generated one after the other
    (an 8 GB card holds one local model at a time).

    Returns {"battle_id", "task", "a": {...}, "b": {...}, "note"}, or
    {"error": ...} when fewer than two models can answer.
    """
    import feedback

    opts = (options or Options()).normalised()
    images = [img for img in (images or []) if img.get("data")]
    policy = cloud.CloudPolicy.from_settings(opts.cloud_mode, opts.allow_docs, opts.allow_paid)
    rag_override = {"always": True, "never": False}.get(opts.rag_mode)
    decision = get_router().route(
        question, force_task=opts.force_task or ("vision" if images else None), policy=policy,
        needs_image=bool(images), rag_override=rag_override,
    )
    if decision["task"] == "system_agent":
        return {"error": "Arena compares model answers; PC actions don't use a model."}
    _record_overrides(question, opts, decision)
    pair = feedback.pick_pair(decision.get("chain") or [], rng)
    if pair is None:
        return {"error": "Arena needs at least two models that can answer this. "
                         "Pull another Ollama model or switch cloud on."}

    sides = {}
    for label, candidate in zip(("a", "b"), pair):
        result = answer(question, options=replace(opts, force_model=candidate["model"]),
                        history=history, images=images, record_signals=False)
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


def _route_for(question: str, opts: Options, images: list) -> tuple[dict[str, Any], Any]:
    policy = cloud.CloudPolicy.from_settings(opts.cloud_mode, opts.allow_docs, opts.allow_paid)
    decision = get_router().route(
        question, force_task=opts.force_task or ("vision" if images else None), policy=policy,
        needs_image=bool(images), rag_override={"always": True, "never": False}.get(opts.rag_mode),
    )
    return decision, policy


def council_recommended(question: str, options: Options | None = None) -> bool:
    """Should [council] auto = "hard" convene the council for this prompt?"""
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
) -> dict[str, Any]:
    """Ask several models, have a judge compare them, return one merged answer.

    Returns {"answer", "members": [...], "judge", "agreements",
    "disagreements", "agreement", "task", "judged", "note"} or {"error"}.
    """
    import concurrent.futures

    import council as council_mod

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

    def ask(candidate: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        try:
            result = answer(question, options=replace(opts, force_model=candidate["model"]),
                            history=history, images=images, record_signals=False)
        except Exception as exc:
            return {"model": candidate["model"], "provider": candidate["provider"],
                    "local": candidate.get("local", True), "answer": "", "error": str(exc),
                    "elapsed": time.monotonic() - started}
        return {"model": result.get("model"), "provider": result.get("provider"),
                "local": result.get("local", True), "answer": result.get("answer", ""),
                "error": None, "elapsed": result.get("elapsed"),
                "sources": result.get("sources") or [], "needs_rag": result.get("needs_rag"),
                "intended": candidate["model"]}

    # Cloud members in parallel while local ones run one at a time.
    cloud_members = [m for m in members if not m.get("local", True)]
    local_members = [m for m in members if m.get("local", True)]
    results: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(cloud_members))) as pool:
        futures = {pool.submit(ask, m): m["model"] for m in cloud_members}
        for m in local_members:
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
    for judge in preferred + [c for c in chain if c not in preferred]:
        if judge.get("provider") == "toolkit":
            continue
        if not judge.get("local", True):
            spec = providers.spec_by_name(judge["model"])
            if (policy.mode == "off" or spec is None
                    or cloud.privacy_block(spec, out["task"], out["needs_rag"], policy)
                    or cloud.provider_status(judge["provider"])["status"] != "ready"):
                continue
        try:
            text = _generate(judge["provider"], judge["model"],
                             council_mod.judge_messages(question, texts), temperature=0.2)
        except Exception as exc:
            if isinstance(exc, cloud.CloudError):
                cloud.record_failure(exc, judge["model"])
            errors.append(f"{judge['model']}: {exc}")
            continue
        if not judge.get("local", True):
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


def local_generate(messages: list[dict[str, str]], task: str = "general",
                   temperature: float = 0.3) -> tuple[str, str]:
    """Generate with the best *local* model for `task`, whatever the cloud
    switch says. For work NEXUS does on its own (digests, flashcards): it
    reads your notes, and nothing leaves this PC unless you asked.

    Returns (text, model); raises RuntimeError when no local model answers.
    """
    avail = providers.availability(include_cloud=False)
    chain = providers.plan(task, "", needs_rag=True, complexity=0.3, avail=avail,
                           policy=cloud.CloudPolicy(mode="off"))
    errors = []
    for candidate in chain:
        if not candidate.spec.is_local:  # belt and braces: policy "off" has none
            continue
        try:
            return _ollama_chat(candidate.model, messages, temperature=temperature), candidate.model
        except Exception as exc:
            errors.append(f"{candidate.model}: {exc}")
    raise RuntimeError("No local model could answer"
                       + (": " + "; ".join(errors[:2]) if errors else " (is Ollama running?)"))


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


def transcribe(
    audio: bytes,
    filename: str = "recording.wav",
    mime: str = "audio/wav",
    options: Options | None = None,
) -> dict[str, Any]:
    """Voice input -> text, following the same cloud switch as answers."""
    import speech

    opts = (options or Options()).normalised()
    policy = cloud.CloudPolicy.from_settings(opts.cloud_mode, opts.allow_docs, opts.allow_paid)
    return speech.transcribe(audio, filename, mime, policy)
