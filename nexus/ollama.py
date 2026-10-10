"""
ollama.py
The one place NEXUS talks to the Ollama daemon.

Model listings are cached for a few seconds because the UI asks on every
rerun; `chat()` streams one /api/chat call and reports timing metrics.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import requests

from nexus import config

TAGS_TTL = 5.0     # installed models change rarely
PS_TTL = 2.0       # which model is resident changes as we generate
LIST_TIMEOUT = 2.0
SHOW_TIMEOUT = 3.0
# 5 s to connect, then at most 120 s of silence between streamed chunks. A long
# answer is fine while tokens keep arriving; a stuck model fails and the chain
# moves on. Loading a 7B model (~30 s here) happens before the first chunk.
CHAT_TIMEOUT = (5, 120)


def url(path: str) -> str:
    return f"{config.OLLAMA_URL}/api/{path.lstrip('/')}"


# ---------------------------------------------------------------------------
# Model listings
# ---------------------------------------------------------------------------


class _Cached:
    """A model-name listing from one endpoint, cached for `ttl` seconds.

    `None` means the daemon didn't answer; that is cached too, so a stopped
    Ollama costs one refused connection per TTL rather than one per call.
    """

    def __init__(self, endpoint: str, ttl: float):
        self.endpoint = endpoint
        self.ttl = ttl
        self._at = -float("inf")
        self._value: list[str] | None = None
        self._lock = threading.Lock()

    def get(self) -> list[str] | None:
        with self._lock:
            if time.monotonic() - self._at < self.ttl:
                return self._value
        value = fetch_names(self.endpoint)
        with self._lock:
            self._value, self._at = value, time.monotonic()
        return value

    def clear(self) -> None:
        with self._lock:
            self._at = -float("inf")


def fetch_names(endpoint: str, timeout: float = LIST_TIMEOUT) -> list[str] | None:
    """Model names from /api/tags or /api/ps, or None if Ollama is unreachable."""
    try:
        response = requests.get(url(endpoint), timeout=timeout)
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", []) if m.get("name")]
    except (requests.RequestException, ValueError, KeyError, TypeError):
        return None


_TAGS = _Cached("tags", TAGS_TTL)
_PS = _Cached("ps", PS_TTL)


def installed_models() -> list[str]:
    """Everything pulled onto this machine ([] when Ollama is down)."""
    return _TAGS.get() or []


def loaded_models() -> list[str]:
    """Models resident in VRAM. Answering with one skips a ~30 s load on an
    8 GB card, which holds one 7B model at a time."""
    return _PS.get() or []


def reachable() -> bool:
    return _TAGS.get() is not None


def clear_cache() -> None:
    _TAGS.clear()
    _PS.clear()


def show(model: str) -> dict[str, Any] | None:
    """Raw /api/show for `model`, or None if Ollama can't say."""
    try:
        response = requests.post(url("show"), json={"model": model}, timeout=SHOW_TIMEOUT)
    except requests.RequestException:
        return None
    if response.status_code != 200:
        return None
    try:
        return response.json()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.S)


@dataclass
class ChatReply:
    text: str
    thinking: str | None
    metrics: dict[str, Any]


def split_inline_thinking(text: str, thinking: str | None) -> tuple[str, str | None]:
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


def chat(
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.7,
    num_ctx: int = config.NUM_CTX,
    think: bool = False,
    on_token: Callable[[str], None] | None = None,
    on_thinking: Callable[[str], None] | None = None,
    images: list[dict[str, str]] | None = None,
) -> ChatReply:
    """One streamed /api/chat call.

    `images` ({"data": <base64>, ...}) are attached to the newest user
    message, which is how Ollama's vision models take them.

    Raises on HTTP errors, error chunks and empty replies. Callback exceptions
    propagate unchanged; the `with` block closes the socket on the way out,
    which is what makes Ollama stop generating.
    """
    if images:
        messages = [dict(m) for m in messages]
        for message in reversed(messages):
            if message["role"] == "user":
                message["images"] = [img["data"] for img in images]
                break
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
    with requests.post(url("chat"), json=payload, timeout=CHAT_TIMEOUT, stream=True) as response:
        if response.status_code >= 400:
            raise RuntimeError(f"{model}: {_error_text(response)}")
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

    text, thinking = split_inline_thinking("".join(parts), "".join(thoughts) or None)
    if not text:
        raise RuntimeError(f"{model} returned an empty response")
    return ChatReply(text, thinking, _metrics(final, sent, first))


def _error_text(response: Any) -> str:
    """Ollama's own error message ("model 'x' not found") beats "HTTP 404"."""
    status = getattr(response, "status_code", "?")
    try:
        body = json.loads(b"".join(response.iter_lines()) or b"{}")
        if isinstance(body, dict) and body.get("error"):
            return f"{body['error']} (HTTP {status})"
    except Exception:
        pass
    return f"HTTP {status}"
