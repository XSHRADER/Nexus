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
        from router import get_installed_ollama_models

        installed_ollama = get_installed_ollama_models()
    if loaded_ollama is None and installed_ollama:
        from router import get_loaded_ollama_models

        loaded_ollama = get_loaded_ollama_models()
    return {
        "ollama": list(installed_ollama or []),
        "ollama_up": bool(installed_ollama),
        "loaded": list(loaded_ollama or []),
    }


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
    for spec in CATALOG:
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
