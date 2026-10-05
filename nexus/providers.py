"""
providers.py
Capability map of every model NEXUS can reach, plus the automatic selection
logic behind it.

The user never picks a model. `plan()` scores the whole catalogue against the
routed task, how hard the request looks, and whether local documents are
involved, then returns an ordered chain of models that are actually reachable
right now. `engine.py` walks that chain top-down until one answers, so a model
that is missing or fails to load degrades instead of failing outright.

Design intent:
  * Everything runs on this PC through Ollama — free, private, and offline.
  * Within that, a specialist beats a generalist: the coder models take coding,
    the chain-of-thought models take planning and deep reasoning, and a model
    already resident in VRAM beats an equal peer that would need loading first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """One reachable model and what it is good at.

    strengths: task -> 0..1 fit. A task absent from the dict means the model is
    not a candidate for it at all.
    quality:   raw capability on hard prompts (matters as complexity rises).
    speed:     responsiveness (matters when the prompt is easy).
    local:     runs on this PC — free, private, no network.
    """

    name: str
    provider: str  # "ollama" | "toolkit"
    strengths: dict[str, float] = field(default_factory=dict)
    quality: float = 0.5
    speed: float = 0.5
    local: bool = True
    note: str = ""
    discovered: bool = False  # profile inferred by discover(), not hand-tuned

    @property
    def is_local(self) -> bool:
        return self.local


CATALOG: list[ModelSpec] = [
    # --- local: general chat -------------------------------------------------
    ModelSpec(
        "llama3.1:8b", "ollama",
        {"general": 0.88, "coding": 0.55, "reasoning": 0.55, "planning": 0.50},
        quality=0.62, speed=0.82, note="balanced local all-rounder",
    ),
    ModelSpec(
        "qwen2.5:7b", "ollama",
        {"general": 0.82, "coding": 0.66, "reasoning": 0.60, "planning": 0.52},
        quality=0.60, speed=0.84,
    ),
    ModelSpec(
        "mistral:7b", "ollama",
        {"general": 0.74, "coding": 0.50, "reasoning": 0.52, "planning": 0.46},
        quality=0.52, speed=0.88,
    ),
    ModelSpec(
        "gemma2:9b", "ollama",
        {"general": 0.80, "coding": 0.55, "reasoning": 0.58, "planning": 0.50},
        quality=0.60, speed=0.76,
    ),
    # --- local: coding -------------------------------------------------------
    ModelSpec(
        "qwen2.5-coder:7b", "ollama",
        {"coding": 0.92, "general": 0.55, "reasoning": 0.50},
        quality=0.70, speed=0.80, note="best local coder",
    ),
    ModelSpec(
        "deepseek-coder:6.7b", "ollama",
        {"coding": 0.86, "general": 0.45},
        quality=0.64, speed=0.82,
    ),
    ModelSpec(
        "codellama:7b", "ollama",
        {"coding": 0.76, "general": 0.42},
        quality=0.56, speed=0.82,
    ),
    # --- local: reasoning ----------------------------------------------------
    ModelSpec(
        "deepseek-r1:7b", "ollama",
        {"reasoning": 0.86, "planning": 0.74, "coding": 0.62, "general": 0.55},
        quality=0.72, speed=0.45, note="local chain-of-thought, slow",
    ),
    ModelSpec(
        "phi4:14b", "ollama",
        {"reasoning": 0.80, "planning": 0.70, "general": 0.70, "coding": 0.60},
        quality=0.74, speed=0.42,
    ),
    # --- local: vision -------------------------------------------------------
    ModelSpec("llava:7b", "ollama", {"vision": 0.78}, quality=0.55, speed=0.70),
    ModelSpec("qwen2.5vl:7b", "ollama", {"vision": 0.82}, quality=0.62, speed=0.66),
    ModelSpec("bakllava:7b", "ollama", {"vision": 0.72}, quality=0.52, speed=0.70),
    # --- local: speech -------------------------------------------------------
    ModelSpec("whisper:small", "ollama", {"speech": 0.85}, quality=0.65, speed=0.70),
    ModelSpec("whisper:base", "ollama", {"speech": 0.72}, quality=0.55, speed=0.85),
]

BY_NAME: dict[str, ModelSpec] = {spec.name: spec for spec in CATALOG}

# Tasks whose answers are built from the user's own documents. Keep these on
# the machine unless nothing local is reachable.
PRIVATE_TASKS = {"system_agent"}

# Preference for staying on this PC, on the same 0..1 scale as fit. Every model
# in the catalogue is local today, so these apply uniformly and do not change
# the ranking — they are kept because they encode the intended policy, and
# become load-bearing again the moment a remote provider is added to CATALOG.
LOCAL_BONUS = 0.12
RAG_LOCAL_BONUS = 0.18

# A 7B model takes ~30s to load into an 8GB card, which holds one at a time.
# So a model already resident answers *much* sooner. Small enough that a real
# specialist (the coder model on a coding task) still displaces it.
WARM_BONUS = 0.10


# ---------------------------------------------------------------------------
# Complexity estimate
# ---------------------------------------------------------------------------

_PLANNING_CUES = re.compile(
    r"\b(plan|planning|roadmap|strategy|strategi[sz]e|step[- ]by[- ]step|"
    r"break (?:it|this) down|outline|milestones?|design a|architect|"
    r"approach|how should i|walk me through|end[- ]to[- ]end)\b",
    re.I,
)
_DEPTH_CUES = re.compile(
    r"\b(compare|trade[- ]?offs?|pros and cons|evaluate|justify|prove|derive|"
    r"in depth|thoroughly|comprehensive|why does|implications?|"
    r"multiple|several (?:steps|options|approaches))\b",
    re.I,
)
_SIMPLE_CUES = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|ok|okay|yes|no|what is|who is|"
    r"define|when is|where is)\b",
    re.I,
)


def estimate_complexity(query: str) -> float:
    """0..1 — how much raw capability this prompt is likely to need.

    Length, planning language, and analytical depth push it up; greetings and
    one-line lookups push it down. This is what decides whether NEXUS reaches
    for the cloud or answers locally.
    """
    text = (query or "").strip()
    if not text:
        return 0.0

    score = 0.0
    words = len(text.split())
    score += min(words / 120.0, 0.35)          # long prompts carry more to juggle
    if _PLANNING_CUES.search(text):
        score += 0.35
    if _DEPTH_CUES.search(text):
        score += 0.25
    if text.count("?") > 1 or re.search(r"\b(and then|after that|also)\b", text, re.I):
        score += 0.10                           # multi-part request
    if _SIMPLE_CUES.match(text) and words < 12:
        score -= 0.30
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


def _normalise(name: str) -> str:
    return name.strip().lower()


def _base(name: str) -> str:
    return _normalise(name).split(":", 1)[0]


def resolve_installed(spec: ModelSpec, installed: list[str]) -> str | None:
    """Return the exact Ollama tag matching `spec`, or None.

    Matches `llama3.1:8b` exactly, and also accepts `llama3.1:latest` /
    `llama3.1` so a user who pulled the default tag still gets routed there.
    """
    wanted = _normalise(spec.name)
    lookup = {_normalise(n): n for n in installed}
    if wanted in lookup:
        return lookup[wanted]
    base = _base(spec.name)
    for norm, original in lookup.items():
        if _base(norm) == base:
            return original
    return None


def availability(
    installed_ollama: list[str] | None = None,
    loaded_ollama: list[str] | None = None,
) -> dict[str, Any]:
    """Snapshot of what NEXUS can reach right now."""
    if installed_ollama is None:
        from nexus.router import get_installed_ollama_models

        installed_ollama = get_installed_ollama_models()
    if loaded_ollama is None and installed_ollama:
        from nexus.router import get_loaded_ollama_models

        loaded_ollama = get_loaded_ollama_models()
    return {
        "ollama": list(installed_ollama or []),
        "ollama_up": bool(installed_ollama),
        "loaded": list(loaded_ollama or []),
    }


# ---------------------------------------------------------------------------
# Capabilities and discovery
# ---------------------------------------------------------------------------

SHOW_URL = "http://localhost:11434/api/show"

# A discovered profile is a guess, so it is scaled down: a hand-tuned catalogue
# entry wins any tie, and a discovered model answers when nothing catalogued
# fits better.
DISCOVERED_SCALE = 0.9

_GENERAL = {"general": 0.75, "coding": 0.50, "reasoning": 0.50, "planning": 0.45}
_CODER = {"coding": 0.80, "general": 0.45}
_REASONER = {"reasoning": 0.78, "planning": 0.68, "coding": 0.55, "general": 0.50}
_VISION = {"vision": 0.72}

_CAPS_CACHE: dict[str, dict[str, Any]] = {}


def _parse_size(text: Any) -> float | None:
    """'8.0B' -> 8.0, '671M' -> 0.671 (billions of parameters)."""
    match = re.match(r"^\s*([\d.]+)\s*([BbMm])", str(text or ""))
    if not match:
        return None
    value = float(match.group(1))
    return value / 1000.0 if match.group(2).lower() == "m" else value


def _size_from_tag(tag: str) -> float | None:
    match = re.search(r":(\d+(?:\.\d+)?)b\b", tag.lower())
    return float(match.group(1)) if match else None


def capabilities(model: str) -> dict[str, Any]:
    """What Ollama reports about `model`: capability tags, context window, size.

    Cached per process once Ollama answers. Returns empty values (not cached)
    when Ollama is unreachable or too old to report them, so callers degrade to
    name-only guesses and default windows.
    """
    if model in _CAPS_CACHE:
        return _CAPS_CACHE[model]
    info: dict[str, Any] = {"caps": set(), "context_length": None, "parameter_size": None}
    try:
        response = requests.post(SHOW_URL, json={"model": model}, timeout=3.0)
    except Exception:
        return info
    if response.status_code == 200:
        data = response.json()
        info["caps"] = {str(c).lower() for c in data.get("capabilities") or []}
        info["parameter_size"] = _parse_size((data.get("details") or {}).get("parameter_size"))
        for key, value in (data.get("model_info") or {}).items():
            if key.endswith(".context_length") and isinstance(value, int):
                info["context_length"] = value
                break
    _CAPS_CACHE[model] = info
    return info


def _infer_strengths(tag: str, caps: set[str]) -> dict[str, float] | None:
    """Task strengths from the model's name, extended by reported capabilities."""
    base = _base(tag)
    if "embed" in base or ("embedding" in caps and "completion" not in caps):
        return None
    if re.search(r"vl\b|llava|vision|moondream", base):
        return dict(_VISION)
    if re.search(r"coder|code", base):
        strengths = dict(_CODER)
    elif re.search(r"r1|qwq|think|reason", base):
        strengths = dict(_REASONER)
    else:
        strengths = dict(_GENERAL)
    # Capability tags add strengths; they never take any away.
    if "thinking" in caps:
        for task, fit in _REASONER.items():
            strengths[task] = max(strengths.get(task, 0.0), fit)
    if "vision" in caps:
        strengths["vision"] = max(strengths.get("vision", 0.0), _VISION["vision"])
    return strengths


def _size_profile(size_b: float | None) -> tuple[float, float]:
    """(quality, speed) from parameter count in billions."""
    if size_b is None or 4 < size_b <= 9:
        return 0.58, 0.80
    if size_b <= 4:
        return 0.45, 0.90
    if size_b <= 15:
        return 0.70, 0.50
    return 0.75, 0.25  # won't fit an 8 GB card; partly runs on the CPU


def discover(installed: list[str]) -> list[ModelSpec]:
    """Profiles for installed models the catalogue doesn't know about."""
    known = {
        resolve_installed(spec, installed)
        for spec in CATALOG
        if spec.provider == "ollama"
    }
    specs: list[ModelSpec] = []
    for tag in installed:
        if tag in known:
            continue
        info = capabilities(tag)
        strengths = _infer_strengths(tag, info["caps"])
        if strengths is None:
            continue
        quality, speed = _size_profile(info["parameter_size"] or _size_from_tag(tag))
        specs.append(
            ModelSpec(
                tag, "ollama",
                {task: round(fit * DISCOVERED_SCALE, 4) for task, fit in strengths.items()},
                quality=quality, speed=speed,
                note="profile inferred from name", discovered=True,
            )
        )
    return specs


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    spec: ModelSpec
    model: str          # exact Ollama tag to call
    score: float
    reason: str


def score_model(
    spec: ModelSpec,
    task: str,
    complexity: float,
    needs_rag: bool,
    warm: bool = False,
) -> float | None:
    """Fit of one model for one request, or None if it can't do the task."""
    fit = spec.strengths.get(task)
    if fit is None:
        return None

    score = fit
    if warm:
        score += WARM_BONUS
    # Hard prompts pay for capability; easy ones pay for responsiveness.
    score += complexity * spec.quality * 0.60
    score += (1.0 - complexity) * spec.speed * 0.35
    if spec.is_local:
        score += LOCAL_BONUS
        if needs_rag:
            score += RAG_LOCAL_BONUS
    return score


def plan(
    task: str,
    query: str = "",
    needs_rag: bool = False,
    complexity: float | None = None,
    avail: dict[str, Any] | None = None,
) -> list[Candidate]:
    """Ordered chain of reachable models for this request, best first."""
    if complexity is None:
        complexity = estimate_complexity(query)
    if avail is None:
        avail = availability()

    installed = avail.get("ollama", [])
    loaded = {_normalise(n) for n in avail.get("loaded", [])}

    candidates: list[Candidate] = []
    for spec in CATALOG + discover(installed):
        if spec.provider != "ollama":
            continue
        resolved = resolve_installed(spec, installed)
        if resolved is None:
            continue
        model_name = resolved
        warm = _normalise(resolved) in loaded

        score = score_model(spec, task, complexity, needs_rag, warm)
        if score is None:
            continue

        candidates.append(
            Candidate(
                spec=spec,
                model=model_name,
                score=score,
                reason=_reason(spec, task, complexity, warm),
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def _reason(spec: ModelSpec, task: str, complexity: float, warm: bool = False) -> str:
    where = "on this PC" if spec.is_local else "in the cloud"
    depth = "complex" if complexity >= 0.5 else "quick"
    detail = f" — {spec.note}" if spec.note else ""
    if warm:
        detail += " (already loaded)"
    return f"{spec.name} {where}: strong on {task}, {depth} request{detail}"


def describe_plan(task: str, query: str = "", needs_rag: bool = False) -> dict[str, Any]:
    """Human-readable view of the automatic choice — used by the UIs."""
    complexity = estimate_complexity(query)
    avail = availability()
    chain = plan(task, query, needs_rag, complexity, avail)
    return {
        "task": task,
        "complexity": round(complexity, 2),
        "chain": [{"model": c.model, "provider": c.spec.provider,
                   "score": round(c.score, 3), "reason": c.reason} for c in chain],
        "available": avail,
    }
