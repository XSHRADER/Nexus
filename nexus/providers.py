"""
providers.py
Capability map of every model NEXUS can reach, plus the automatic selection
logic behind it.

`plan()` scores every installed model against the routed task, how hard the
request looks, and whether it is already loaded, then returns an ordered chain
of models that are actually reachable right now. `engine.py` walks that chain
top-down until one answers, so a model that is missing or fails to load
degrades instead of failing outright.

Design intent:
  * Everything runs on this PC through Ollama — free, private, and offline.
  * A specialist beats a generalist: the coder models take coding, the
    chain-of-thought models take planning and deep reasoning, and a model
    already resident in VRAM beats an equal peer that would need loading first.
  * Models outside the hand-tuned catalogue still take part, with a profile
    inferred from their name, size and reported capabilities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from nexus import ollama

# Tasks a text prompt can be routed to. Image and audio models are recognised
# (so they are never offered for text) but there is no attachment input yet.
TEXT_TASKS = ("general", "coding", "reasoning", "planning")

# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """One model and what it is good at.

    strengths: task -> 0..1 fit. A task absent from the dict means the model is
    not a candidate for it at all.
    quality:   raw capability on hard prompts (matters as complexity rises).
    speed:     responsiveness (matters when the prompt is easy).
    """

    name: str
    provider: str  # "ollama" | "toolkit"
    strengths: dict[str, float] = field(default_factory=dict)
    quality: float = 0.5
    speed: float = 0.5
    note: str = ""
    discovered: bool = False  # profile inferred by discover(), not hand-tuned


CATALOG: list[ModelSpec] = [
    # --- general chat --------------------------------------------------------
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
    # --- coding --------------------------------------------------------------
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
    # --- reasoning -----------------------------------------------------------
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
    # --- vision (recognised so they are kept out of text routing) ------------
    ModelSpec("llava:7b", "ollama", {"vision": 0.78}, quality=0.55, speed=0.70),
    ModelSpec("qwen2.5vl:7b", "ollama", {"vision": 0.82}, quality=0.62, speed=0.66),
]

BY_NAME: dict[str, ModelSpec] = {spec.name: spec for spec in CATALOG}

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
    one-line lookups push it down. High complexity favours the stronger,
    slower models; low complexity favours the fast ones.
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


def _size_from_tag(tag: str) -> float | None:
    """`llama3.1:8b-instruct-q4_K_M` -> 8.0; `llama3.1:latest` -> None."""
    match = re.search(r":(\d+(?:\.\d+)?)b\b", tag.lower())
    return float(match.group(1)) if match else None


def resolve_installed(spec: ModelSpec, installed: list[str]) -> str | None:
    """Return the exact Ollama tag matching `spec`, or None.

    Matches `llama3.1:8b` exactly, and also accepts `llama3.1:latest`,
    `llama3.1` or `llama3.1:8b-instruct-q4_K_M` -- but not `llama3.1:70b`:
    a different size is a different model, with a different speed profile.
    """
    wanted = _normalise(spec.name)
    lookup = {_normalise(n): n for n in installed}
    if wanted in lookup:
        return lookup[wanted]
    base, size = _base(spec.name), _size_from_tag(spec.name)
    for norm, original in lookup.items():
        if _base(norm) != base:
            continue
        other = _size_from_tag(norm)
        if size is None or other is None or other == size:
            return original
    return None


def availability(
    installed_ollama: list[str] | None = None,
    loaded_ollama: list[str] | None = None,
) -> dict[str, Any]:
    """Snapshot of what NEXUS can reach right now."""
    if installed_ollama is None:
        installed_ollama = ollama.installed_models()
    if loaded_ollama is None:
        loaded_ollama = ollama.loaded_models() if installed_ollama else []
    return {
        "ollama": list(installed_ollama),
        "ollama_up": bool(installed_ollama),
        "loaded": list(loaded_ollama),
    }


# ---------------------------------------------------------------------------
# Capabilities and discovery
# ---------------------------------------------------------------------------

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


def capabilities(model: str) -> dict[str, Any]:
    """What Ollama reports about `model`: capability tags, context window, size.

    Cached per process once Ollama answers. Returns empty values (not cached)
    when Ollama is unreachable or too old to report them, so callers degrade to
    name-only guesses and default windows.
    """
    if model in _CAPS_CACHE:
        return _CAPS_CACHE[model]
    info: dict[str, Any] = {"caps": set(), "context_length": None, "parameter_size": None}
    data = ollama.show(model)
    if data is None:
        return info
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
    known = {resolve_installed(spec, installed) for spec in CATALOG}
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


def score_model(spec: ModelSpec, task: str, complexity: float, warm: bool = False) -> float | None:
    """Fit of one model for one request, or None if it can't do the task."""
    fit = spec.strengths.get(task)
    if fit is None:
        return None
    score = fit + (WARM_BONUS if warm else 0.0)
    # Hard prompts pay for capability; easy ones pay for responsiveness.
    score += complexity * spec.quality * 0.60
    score += (1.0 - complexity) * spec.speed * 0.35
    return score


def plan(
    task: str,
    query: str = "",
    complexity: float | None = None,
    avail: dict[str, Any] | None = None,
) -> list[Candidate]:
    """Ordered chain of reachable models for this request, best first.

    Each installed model appears at most once: two catalogue entries can
    resolve to the same tag (`llava:7b` and a bare `llava`), and retrying the
    same model after it failed only doubles the wait.
    """
    if complexity is None:
        complexity = estimate_complexity(query)
    if avail is None:
        avail = availability()

    installed = avail.get("ollama", [])
    loaded = {_normalise(n) for n in avail.get("loaded", [])}

    best: dict[str, Candidate] = {}
    for spec in CATALOG + discover(installed):
        resolved = resolve_installed(spec, installed)
        if resolved is None:
            continue
        warm = _normalise(resolved) in loaded
        score = score_model(spec, task, complexity, warm)
        if score is None:
            continue
        if resolved not in best or score > best[resolved].score:
            best[resolved] = Candidate(spec, resolved, score, _reason(spec, task, complexity, warm))

    return sorted(best.values(), key=lambda c: c.score, reverse=True)


def _reason(spec: ModelSpec, task: str, complexity: float, warm: bool = False) -> str:
    depth = "complex" if complexity >= 0.5 else "quick"
    detail = f" — {spec.note}" if spec.note else ""
    if warm:
        detail += " (already loaded)"
    return f"{spec.name}: strong on {task}, {depth} request{detail}"
