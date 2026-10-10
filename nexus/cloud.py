"""
cloud.py
Everything NEXUS knows about cloud providers, short of talking to them:
which ones exist, whether each is usable right now, and whether a given
request is allowed to leave this PC at all.

Three layers decide whether a cloud model may answer:

1. Reachability -- the provider has a key, its key hasn't been rejected,
   it isn't cooling down after a rate limit or outage, and today's request
   count is under its daily limit.
2. Policy -- the cloud switch (off / hard / allowed), whether paid models
   are allowed, and the privacy rules: questions that use your documents
   stay local unless you allow them out, and PC actions always stay local.
3. Scoring -- providers.plan() ranks whatever survives 1 and 2 against the
   local models.

The talking itself is in cloud_client.py.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from nexus.config import CLOUD_MODES, PROJECT_DIR, get_settings

CATALOG_FILE = PROJECT_DIR / "cloud_models.toml"

# OpenAI-style base URLs (the client appends /chat/completions, /models,
# /audio/transcriptions).
PROVIDERS: dict[str, dict[str, str]] = {
    "gemini": {
        "label": "Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "key_env": "GEMINI_API_KEY",
    },
    "groq": {
        "label": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
    },
    "openrouter": {
        "label": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
    },
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "key_env": "DEEPSEEK_API_KEY",
    },
    "mistral": {
        "label": "Mistral",
        "base_url": "https://api.mistral.ai/v1",
        "key_env": "MISTRAL_API_KEY",
    },
}

# Tasks that never leave the PC whatever the settings say.
PRIVATE_TASKS = {"system_agent"}

# An outage clears faster than a rate limit, so it is retried sooner.
OFFLINE_COOLDOWN_SECONDS = 60


# ---------------------------------------------------------------------------
# Errors -- typed so the engine can react to *why* a call failed
# ---------------------------------------------------------------------------


class CloudError(RuntimeError):
    """A cloud call failed. The engine skips to the next model."""

    def __init__(self, provider: str, message: str):
        super().__init__(f"{provider}: {message}")
        self.provider = provider


class AuthError(CloudError):
    """Key missing or rejected (401/403). The provider is off until the key changes."""


class RateLimited(CloudError):
    """429. The provider cools down for `retry_after` seconds (or the default)."""

    def __init__(self, provider: str, message: str, retry_after: float | None = None):
        super().__init__(provider, message)
        self.retry_after = retry_after


class ModelNotFound(CloudError):
    """404 for a model name. That model is skipped; the provider stays up."""

    def __init__(self, provider: str, message: str, model: str = ""):
        super().__init__(provider, message)
        self.model = model


class Offline(CloudError):
    """Connection failure, timeout or 5xx. Short cooldown."""


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------


def provider_config(provider: str) -> dict[str, str]:
    base = dict(PROVIDERS.get(provider, {}))
    base.update({k: str(v) for k, v in get_settings().provider_overrides.get(provider, {}).items()})
    return base


def api_key(provider: str) -> str:
    env = provider_config(provider).get("key_env", "")
    return os.environ.get(env, "").strip() if env else ""


def _fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:12] if key else ""


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------


def _load_catalog_entries(path: Path = CATALOG_FILE) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "rb") as fh:
        return list(tomllib.load(fh).get("model", []))


def merged_entries() -> list[dict[str, Any]]:
    """cloud_models.toml, with nexus.toml's [[cloud.models]] added or
    replacing same-named entries."""
    by_name: dict[str, dict[str, Any]] = {}
    for entry in _load_catalog_entries() + list(get_settings().extra_models):
        if entry.get("name") and entry.get("provider") in PROVIDERS:
            by_name[str(entry["name"])] = dict(entry)
    return list(by_name.values())


@lru_cache(maxsize=1)
def cloud_specs() -> tuple:
    """ModelSpecs for every catalogued cloud model (reachable or not)."""
    from nexus.providers import ModelSpec

    specs = []
    for e in merged_entries():
        if e.get("enabled", True) is False:
            continue
        specs.append(
            ModelSpec(
                name=str(e["name"]),
                provider=str(e["provider"]),
                strengths={str(k): float(v) for k, v in (e.get("strengths") or {}).items()},
                quality=float(e.get("quality", 0.7)),
                speed=float(e.get("speed", 0.7)),
                note=str(e.get("note", "")),
                cost=str(e.get("cost", "paid")),
                modalities=frozenset(e.get("modalities") or ["text"]),
                context=int(e.get("context", 32000)),
            )
        )
    return tuple(specs)


# ---------------------------------------------------------------------------
# Live provider state (in memory; resets when NEXUS restarts)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_cooldowns: dict[str, tuple[float, str]] = {}      # provider -> (until, why)
_bad_keys: dict[str, str] = {}                     # provider -> rejected key fingerprint
_missing_models: set[tuple[str, str]] = set()      # (provider, model) that 404'd


def reset_state() -> None:
    with _lock:
        _cooldowns.clear()
        _bad_keys.clear()
        _missing_models.clear()


def record_failure(exc: CloudError, model: str = "") -> None:
    """Remember a failure so the next request skips the broken provider."""
    now = time.time()
    with _lock:
        if isinstance(exc, AuthError):
            _bad_keys[exc.provider] = _fingerprint(api_key(exc.provider))
        elif isinstance(exc, RateLimited):
            wait = exc.retry_after or get_settings().cooldown_minutes * 60
            _cooldowns[exc.provider] = (now + wait, "rate-limited")
        elif isinstance(exc, ModelNotFound):
            _missing_models.add((exc.provider, exc.model or model))
        elif isinstance(exc, Offline):
            _cooldowns[exc.provider] = (now + OFFLINE_COOLDOWN_SECONDS, "unreachable")


def record_success(provider: str, chars_in: int = 0, chars_out: int = 0, store=None) -> None:
    """Count a completed cloud request toward today's limit."""
    if store is None:
        from nexus.store import get_store

        store = get_store()
    store.record_usage(provider, chars_in, chars_out)


def is_missing(provider: str, model: str) -> bool:
    return (provider, model) in _missing_models


def provider_status(provider: str, store=None) -> dict[str, Any]:
    """{"status": ready | no_key | invalid_key | cooling_down | limit_reached,
        "detail": str, "used": int, "limit": int}"""
    settings = get_settings()
    limit = int(settings.daily_limits.get(provider, 0))
    key = api_key(provider)
    if not key:
        env = provider_config(provider).get("key_env", "")
        return {"status": "no_key", "detail": f"set {env} in .env", "used": 0, "limit": limit}
    with _lock:
        bad = _bad_keys.get(provider)
        cooldown = _cooldowns.get(provider)
    if bad and bad == _fingerprint(key):
        return {"status": "invalid_key", "detail": "the provider rejected this key",
                "used": 0, "limit": limit}

    if store is None:
        from nexus.store import get_store

        store = get_store()
    used = store.usage_today(provider)
    if cooldown and cooldown[0] > time.time():
        left = int(cooldown[0] - time.time())
        return {"status": "cooling_down", "detail": f"{cooldown[1]}; retrying in {left}s",
                "used": used, "limit": limit}
    if limit and used >= limit:
        return {"status": "limit_reached", "detail": f"{used}/{limit} requests today",
                "used": used, "limit": limit}
    return {"status": "ready", "detail": f"{used}/{limit or '∞'} requests today",
            "used": used, "limit": limit}


def statuses(store=None) -> dict[str, dict[str, Any]]:
    return {p: provider_status(p, store) for p in PROVIDERS}


def ready_providers(store=None) -> set[str]:
    return {p for p, s in statuses(store).items() if s["status"] == "ready"}


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CloudPolicy:
    mode: str = "off"                 # off | hard | allowed
    allow_docs: bool = False
    allow_paid: bool = False
    hard_threshold: float = 0.6

    @classmethod
    def from_settings(cls, mode: str | None = None, allow_docs: bool | None = None,
                      allow_paid: bool | None = None) -> CloudPolicy:
        s = get_settings()
        chosen = (mode or s.cloud_mode).lower()
        return cls(
            mode=chosen if chosen in CLOUD_MODES else "off",
            allow_docs=s.allow_docs_to_cloud if allow_docs is None else bool(allow_docs),
            allow_paid=s.allow_paid if allow_paid is None else bool(allow_paid),
            hard_threshold=s.hard_threshold,
        )


def privacy_block(spec, task: str, needs_rag: bool, policy: CloudPolicy) -> str | None:
    """Why `spec` may not see this request, or None if it may.

    These are the rules that hold even for a model you pinned yourself:
    pinning picks the model, it doesn't widen what may leave the PC.
    """
    if spec.is_local:
        return None
    if task in PRIVATE_TASKS:
        return "PC actions always stay on this PC"
    if needs_rag and not policy.allow_docs:
        return "questions about your documents stay on this PC (Send documents to cloud is off)"
    if spec.cost == "paid" and not policy.allow_paid:
        return "paid models are off (Allow paid models)"
    return None


def cloud_candidate_ok(spec, task: str, needs_rag: bool, policy: CloudPolicy,
                       ready: set[str]) -> bool:
    """Policy + reachability for automatic routing (the cloud switch applies)."""
    if spec.is_local:
        return True
    if policy.mode == "off":
        return False
    if spec.provider not in ready or is_missing(spec.provider, spec.name):
        return False
    return privacy_block(spec, task, needs_rag, policy) is None


# ---------------------------------------------------------------------------
# Model-name check (Diagnostics)
# ---------------------------------------------------------------------------


def verify_models() -> dict[str, dict[str, Any]]:
    """Ask each keyed provider what it serves; report configured names it doesn't.

    {provider: {"ok": [...], "missing": [...], "error": str|None}}
    """
    from nexus import cloud_client

    report: dict[str, dict[str, Any]] = {}
    for provider in PROVIDERS:
        names = [s.name for s in cloud_specs() if s.provider == provider]
        if not names or not api_key(provider):
            continue
        try:
            served = set(cloud_client.list_models(provider))
        except CloudError as exc:
            report[provider] = {"ok": [], "missing": [], "error": str(exc)}
            continue
        # Gemini lists "models/<id>"; compare on the bare id.
        served |= {n.split("/", 1)[1] for n in served if n.startswith("models/")}
        report[provider] = {
            "ok": [n for n in names if n in served],
            "missing": [n for n in names if n not in served],
            "error": None,
        }
    return report
