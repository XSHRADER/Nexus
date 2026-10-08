"""
cloud_client.py
One client for every provider that speaks the OpenAI-style API: Gemini,
Groq, OpenRouter, DeepSeek and Mistral. Adding another such provider is an
entry in cloud.PROVIDERS, not new code.

Uses `requests` (already a dependency) and turns every failure into one of
cloud.py's typed errors, so the engine can tell "rate-limited, try again
later" from "bad key" from "no internet" and fall through to the next model.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Iterable

import requests

from cloud import (
    AuthError,
    CloudError,
    ModelNotFound,
    Offline,
    RateLimited,
    api_key,
    provider_config,
)

TIMEOUT = 120          # seconds; generation can be slow on hard prompts
LIST_TIMEOUT = 15


def _base_url(provider: str) -> str:
    url = provider_config(provider).get("base_url", "").rstrip("/")
    if not url:
        raise CloudError(provider, "no base_url configured")
    return url


def _headers(provider: str) -> dict[str, str]:
    key = api_key(provider)
    if not key:
        raise AuthError(provider, "no API key set")
    headers = {"Authorization": f"Bearer {key}"}
    if provider == "openrouter":
        # OpenRouter's recommended attribution headers; harmless elsewhere.
        headers["X-Title"] = "NEXUS"
    return headers


def _retry_after(response: requests.Response) -> float | None:
    value = response.headers.get("Retry-After")
    try:
        return float(value) if value else None
    except ValueError:
        return None


def _error_text(response: requests.Response) -> str:
    try:
        body = response.json()
        err = body.get("error", body)
        if isinstance(err, dict):
            return str(err.get("message") or err)[:300]
        return str(err)[:300]
    except ValueError:
        return (response.text or response.reason or "")[:300]


def raise_for(provider: str, response: requests.Response, model: str = "") -> None:
    """Map an HTTP failure to the typed error the engine reacts to."""
    status = response.status_code
    if status < 400:
        return
    detail = _error_text(response)
    if status in (401, 403):
        raise AuthError(provider, f"key rejected ({status}): {detail}")
    if status == 429:
        raise RateLimited(provider, f"rate-limited: {detail}", _retry_after(response))
    if status == 404:
        raise ModelNotFound(provider, f"model not found: {detail}", model)
    if status >= 500:
        raise Offline(provider, f"server error {status}: {detail}")
    raise CloudError(provider, f"request failed ({status}): {detail}")


# ---------------------------------------------------------------------------
# Message shapes
# ---------------------------------------------------------------------------


def to_openai_messages(
    messages: list[dict[str, str]], images: list[dict[str, str]] | None = None
) -> list[dict[str, Any]]:
    """NEXUS messages -> OpenAI-style. Images ride on the last user turn as
    `image_url` parts holding a base64 data URI."""
    out: list[dict[str, Any]] = [{"role": m["role"], "content": m["content"]} for m in messages]
    if images:
        for i in range(len(out) - 1, -1, -1):
            if out[i]["role"] == "user":
                parts: list[dict[str, Any]] = [{"type": "text", "text": out[i]["content"]}]
                for img in images:
                    uri = f"data:{img.get('mime', 'image/png')};base64,{img['data']}"
                    parts.append({"type": "image_url", "image_url": {"url": uri}})
                out[i]["content"] = parts
                break
    return out


def iter_sse_text(lines: Iterable[bytes | str]) -> Iterable[str]:
    """Text deltas from an OpenAI-style server-sent-event stream.

    Tolerates blank keep-alive lines, `:` comments, and non-JSON noise, and
    stops at `data: [DONE]`.
    """
    for raw in lines:
        if not raw:
            continue
        line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(chunk, dict) and chunk.get("error"):
            raise CloudError("stream", str(chunk["error"])[:300])
        for choice in chunk.get("choices") or []:
            text = (choice.get("delta") or {}).get("content")
            if text:
                yield text


# ---------------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------------


def chat(
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    on_token: Callable[[str], None] | None = None,
    images: list[dict[str, str]] | None = None,
    timeout: int = TIMEOUT,
) -> str:
    url = f"{_base_url(provider)}/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": to_openai_messages(messages, images),
        "temperature": float(temperature),
        "stream": on_token is not None,
    }
    headers = _headers(provider)
    try:
        if on_token is None:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
            raise_for(provider, response, model)
            choices = response.json().get("choices") or [{}]
            content = (choices[0].get("message") or {}).get("content") or ""
            if isinstance(content, list):  # some providers return parts
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            text = content
        else:
            parts: list[str] = []
            with requests.post(url, json=payload, headers=headers, timeout=timeout,
                               stream=True) as response:
                raise_for(provider, response, model)
                for token in iter_sse_text(response.iter_lines()):
                    parts.append(token)
                    on_token(token)
            text = "".join(parts)
    except requests.RequestException as exc:
        raise Offline(provider, f"unreachable: {exc.__class__.__name__}") from exc

    text = text.strip()
    if not text:
        raise CloudError(provider, f"{model} returned an empty response")
    return text


def list_models(provider: str) -> list[str]:
    try:
        response = requests.get(f"{_base_url(provider)}/models", headers=_headers(provider),
                                timeout=LIST_TIMEOUT)
        raise_for(provider, response)
        body = response.json()
    except requests.RequestException as exc:
        raise Offline(provider, f"unreachable: {exc.__class__.__name__}") from exc
    items = body.get("data") or body.get("models") or []
    return [str(m.get("id") or m.get("name")) for m in items if isinstance(m, dict)]


def transcribe(
    provider: str,
    model: str,
    audio: bytes,
    filename: str = "audio.wav",
    mime: str = "audio/wav",
    timeout: int = TIMEOUT,
) -> str:
    """Speech to text through the OpenAI-style /audio/transcriptions endpoint."""
    url = f"{_base_url(provider)}/audio/transcriptions"
    try:
        response = requests.post(
            url,
            headers=_headers(provider),
            files={"file": (filename, audio, mime)},
            data={"model": model, "response_format": "json"},
            timeout=timeout,
        )
        raise_for(provider, response, model)
        text = (response.json().get("text") or "").strip()
    except requests.RequestException as exc:
        raise Offline(provider, f"unreachable: {exc.__class__.__name__}") from exc
    if not text:
        raise CloudError(provider, "no speech recognised in the recording")
    return text
