"""
config.py
One place for NEXUS settings.

Two sources, both optional:
  * `.env`        secrets (API keys). Never committed -- see `.env.example`.
  * `nexus.toml`  behaviour (memory size, database path, later cloud rules).
                  Copy `nexus.toml.example` to start.

A missing file means "use the defaults", and the defaults reproduce the
behaviour NEXUS had before either file existed, so a fresh clone needs neither.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent
ENV_FILE = PROJECT_DIR / ".env"
TOML_FILE = PROJECT_DIR / "nexus.toml"


def load_env(path: Path = ENV_FILE) -> dict[str, str]:
    """Read `KEY=value` lines into os.environ without overriding real env vars.

    Deliberately tiny: comments, blank lines, optional `export ` and matching
    quotes. Returns what the file contained so callers can test it.
    """
    found: dict[str, str] = {}
    if not path.exists():
        return found
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not key:
            continue
        found[key] = value
        os.environ.setdefault(key, value)
    return found


@dataclass(frozen=True)
class Settings:
    # -- conversation memory --------------------------------------------------
    # How many earlier messages (user + assistant) go back to the model, and
    # a character cap on them. ~4 characters per token, so 8000 chars is about
    # 2,000 tokens: enough for a real follow-up without crowding out RAG
    # context in an 8k window.
    history_messages: int = 8
    history_chars: int = 8000
    # -- storage --------------------------------------------------------------
    db_path: Path = PROJECT_DIR / "nexus.db"

    # -- cloud ----------------------------------------------------------------
    # "off": never leave this PC (the default, and NEXUS's behaviour before
    # cloud support existed). "hard": cloud only for hard prompts, or tasks no
    # local model can do. "allowed": cloud competes on every prompt, but easy
    # ones still go local first.
    cloud_mode: str = "off"
    # Questions answered from your documents send those documents' text to
    # the model. Off by default: they stay on this PC.
    allow_docs_to_cloud: bool = False
    # Models billed per token with no free tier (OpenRouter auto).
    allow_paid: bool = False
    # Difficulty (0..1) at which a prompt counts as "hard".
    hard_threshold: float = 0.6
    # After a rate limit or outage, skip that provider for this long.
    cooldown_minutes: float = 5.0
    # Requests per provider per day. A provider at its limit is skipped.
    daily_limits: dict[str, int] = field(default_factory=dict)
    # Per-provider overrides, e.g. {"gemini": {"base_url": "..."}}.
    provider_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Extra or replacement cloud models, same shape as cloud_models.toml.
    extra_models: list[dict[str, Any]] = field(default_factory=list)

    # -- speech ---------------------------------------------------------------
    speech_provider: str = "groq"
    speech_model: str = "whisper-large-v3-turbo"
    # Used when cloud is off (or the cloud call fails) and faster-whisper is
    # installed: tiny / base / small / medium / large-v3.
    local_whisper_model: str = "small"

    # -- truth check ----------------------------------------------------------
    # "auto": the NLI model when it can be loaded, else the keyword checker.
    # "nli" / "keyword" force one. "off" disables the check.
    truth_method: str = "auto"
    truth_model: str = "cross-encoder/nli-deberta-v3-small"
    # Probability (0..1) needed to call a sentence supported / contradicted.
    support_threshold: float = 0.5
    contradict_threshold: float = 0.5
    truth_max_claims: int = 20
    evidence_per_claim: int = 3

    # -- learned router -------------------------------------------------------
    # "auto": use the trained router (models/current.json) when there is one,
    # else the keyword rules. "rules" / "learned" force one.
    router_mode: str = "auto"
    # Below this probability the learned task guess defers to the rules.
    router_min_confidence: float = 0.5
    # P(needs your documents) at which retrieval switches on.
    docs_threshold: float = 0.5
    # P(needs a strong model) at which a prompt counts as hard for cloud.
    strong_threshold: float = 0.5
    # How much your Arena results move a model's score (0 = ignore them).
    preference_weight: float = 0.10

    # -- model council --------------------------------------------------------
    council_members: int = 3
    # "off": only when you switch Council on. "hard": also convene it by
    # itself for prompts the router marks as hard (slower: 3 answers + judge).
    council_auto: str = "off"
    # Judge model; empty = the strongest model allowed for the request.
    council_judge: str = ""

    # -- background brain -----------------------------------------------------
    brain_enabled: bool = True
    # How often to look for new or changed files.
    brain_scan_seconds: float = 60.0
    # A digest is written when the last one is this many days old (0 = never).
    digest_days: float = 7.0
    # Make flashcards from new and changed notes.
    study_enabled: bool = True
    cards_per_file: int = 6
    # Extra folders whose notes feed the digest and flashcards. Only
    # documents/ is indexed for answering questions.
    watch_folders: list[str] = field(default_factory=list)

    raw: dict[str, Any] | None = None  # the parsed toml, for later phases


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "rb") as fh:
        return tomllib.load(fh)


CLOUD_MODES = ("off", "hard", "allowed")

DEFAULT_DAILY_LIMITS = {
    "gemini": 200,
    "groq": 500,
    "openrouter": 50,
    "deepseek": 100,
    "mistral": 100,
}


def _bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return default


def load_settings(toml_path: Path = TOML_FILE, env_path: Path = ENV_FILE) -> Settings:
    load_env(env_path)
    data = _read_toml(toml_path)
    memory = data.get("memory", {})
    storage = data.get("storage", {})
    cloud = data.get("cloud", {})
    speech = data.get("speech", {})
    truth = data.get("truth", {})
    router = data.get("router", {})
    council = data.get("council", {})
    brain = data.get("brain", {})

    mode = str(cloud.get("mode", Settings.cloud_mode)).strip().lower()
    if mode not in CLOUD_MODES:
        mode = "off"
    limits = dict(DEFAULT_DAILY_LIMITS)
    limits.update({str(k): int(v) for k, v in (cloud.get("daily_limits") or {}).items()})

    db_path = Path(storage.get("db_path", Settings.db_path))
    if not db_path.is_absolute():
        db_path = PROJECT_DIR / db_path

    return Settings(
        history_messages=max(0, int(memory.get("history_messages", Settings.history_messages))),
        history_chars=max(0, int(memory.get("history_chars", Settings.history_chars))),
        db_path=db_path,
        cloud_mode=mode,
        allow_docs_to_cloud=_bool(cloud.get("allow_docs"), False),
        allow_paid=_bool(cloud.get("allow_paid"), False),
        hard_threshold=float(cloud.get("hard_threshold", Settings.hard_threshold)),
        cooldown_minutes=float(cloud.get("cooldown_minutes", Settings.cooldown_minutes)),
        daily_limits=limits,
        provider_overrides={str(k): dict(v) for k, v in (cloud.get("providers") or {}).items()},
        extra_models=list(cloud.get("models") or []),
        speech_provider=str(speech.get("provider", Settings.speech_provider)),
        speech_model=str(speech.get("model", Settings.speech_model)),
        local_whisper_model=str(speech.get("local_model", Settings.local_whisper_model)),
        truth_method=str(truth.get("method", Settings.truth_method)).lower(),
        truth_model=str(truth.get("model", Settings.truth_model)),
        support_threshold=float(truth.get("support_threshold", Settings.support_threshold)),
        contradict_threshold=float(truth.get("contradict_threshold",
                                             Settings.contradict_threshold)),
        truth_max_claims=max(1, int(truth.get("max_claims", Settings.truth_max_claims))),
        evidence_per_claim=max(1, int(truth.get("evidence_per_claim",
                                                Settings.evidence_per_claim))),
        router_mode=str(router.get("mode", Settings.router_mode)).lower(),
        router_min_confidence=float(router.get("min_confidence", Settings.router_min_confidence)),
        docs_threshold=float(router.get("docs_threshold", Settings.docs_threshold)),
        strong_threshold=float(router.get("strong_threshold", Settings.strong_threshold)),
        preference_weight=float(router.get("preference_weight", Settings.preference_weight)),
        council_members=max(2, min(5, int(council.get("members", Settings.council_members)))),
        council_auto=str(council.get("auto", Settings.council_auto)).lower(),
        council_judge=str(council.get("judge", Settings.council_judge)),
        brain_enabled=_bool(brain.get("enabled"), True),
        brain_scan_seconds=max(5.0, float(brain.get("scan_seconds", Settings.brain_scan_seconds))),
        digest_days=max(0.0, float(brain.get("digest_days", Settings.digest_days))),
        study_enabled=_bool(brain.get("study"), True),
        cards_per_file=max(1, min(20, int(brain.get("cards_per_file", Settings.cards_per_file)))),
        watch_folders=[str(f) for f in brain.get("watch_folders") or []],
        raw=data,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
