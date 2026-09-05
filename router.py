"""
router.py
Task router for NEXUS AI: classifies a query as general chat, coding,
reasoning, planning, vision, speech, or a PC action, decides when local
document retrieval should be included, and hands the task to `providers.plan()`
which picks the model automatically.

The user never selects a model anywhere in the app.
"""

import json
import re
import time
from pathlib import Path
from typing import Any
import requests

import providers
from embeddings import get_sentence_transformer

_OLLAMA_CACHE: dict[str, Any] = {"ts": 0.0, "models": []}
_OLLAMA_TTL = 5.0  # seconds -- route() runs per keystroke-ish; don't hammer the daemon
_LOADED_CACHE: dict[str, Any] = {"ts": 0.0, "models": []}
_LOADED_TTL = 2.0  # shorter: which model is resident changes as we generate


def _ollama_models(endpoint: str, cache: dict[str, Any], ttl: float) -> list[str]:
    now = time.monotonic()
    if now - cache["ts"] < ttl:
        return cache["models"]
    models: list[str] = []
    try:
        response = requests.get(f"http://localhost:11434/api/{endpoint}", timeout=2.0)
        if response.status_code == 200:
            models = [m["name"] for m in response.json().get("models", [])]
    except Exception:
        pass
    cache.update(ts=now, models=models)
    return models


def get_installed_ollama_models() -> list[str]:
    """Everything pulled onto this machine."""
    return _ollama_models("tags", _OLLAMA_CACHE, _OLLAMA_TTL)


def get_loaded_ollama_models() -> list[str]:
    """Models currently resident in VRAM — answering with one of these skips a
    ~30s load on an 8GB card, which only holds one 7B model at a time."""
    return _ollama_models("ps", _LOADED_CACHE, _LOADED_TTL)

PROJECT_DIR = Path(__file__).resolve().parent
LOG_FILE = PROJECT_DIR / "router_logs.jsonl"


class DecisionLogger:
    def __init__(self, log_file: str | Path = LOG_FILE):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: dict[str, Any]) -> None:
        with open(self.log_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


class TaskRouter:
    """Simple rule-based router with lightweight semantic fallback."""

    TASKS = ("general", "coding", "reasoning", "planning", "vision", "speech", "system_agent")

    # Every model that can serve each task, best-typical-fit first. Built from
    # the capability catalogue so there is exactly one place to add a model.
    # This is display/reference only — the live choice comes from
    # `providers.plan()`, which also weighs availability and prompt difficulty.
    MODEL_MAP = {
        task: [
            spec.name
            for spec in sorted(
                (s for s in providers.CATALOG if task in s.strengths),
                key=lambda s: s.strengths[task],
                reverse=True,
            )
        ]
        for task in TASKS
    }
    MODEL_MAP["system_agent"] = ["pc-toolkit"]

    CATEGORY_EXAMPLES = {
        "general": [
            "Summarize this project in simple terms.",
            "Explain the purpose of a local AI assistant.",
        ],
        "coding": [
            "Fix a Python TypeError in a function.",
            "Write a debugging plan for a failing API call.",
        ],
        "reasoning": [
            "Compare the trade-offs between caching and message queues.",
            "Analyze the pros and cons of a multi-step decision.",
        ],
        "planning": [
            "Make me a step by step plan to finish this project.",
            "Draft a roadmap for the next two weeks of work.",
            "Break this task down into ordered steps I can follow.",
            "How should I approach building this from scratch?",
        ],
        "vision": [
            "Describe what is in this image.",
            "Read the chart and explain the trends.",
        ],
        "speech": [
            "Transcribe this spoken conversation.",
            "Convert the audio note into a written summary.",
        ],
        "system_agent": [
            "Organize my downloads folder by file types.",
            "Sort the files on my desktop into folders.",
            "Tidy up this folder and clean out empty directories.",
            "Undo the last folder organization.",
            "Find duplicate files taking up space on my PC.",
            "Scan directory for large files over 100MB.",
            "Analyze disk usage and file clutter in a folder.",
        ],
    }

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embed_model = get_sentence_transformer(embedding_model)
        self.logger = DecisionLogger()
        # Pre-embed the category exemplars once; reused on every query.
        self.category_vectors = {
            label: self.embed_model.encode(examples, normalize_embeddings=True)
            for label, examples in self.CATEGORY_EXAMPLES.items()
        }

    def _rule_scores(self, query: str) -> dict[str, float]:
        text = query.lower()
        scores = {name: 0.0 for name in self.TASKS}
        # Chat is the safe default: without it, a short greeting has no keyword
        # signal at all and embedding noise can hand it to a vision model.
        scores["general"] += 1.2
        if re.match(r"^\s*(hi|hey|hello|yo|sup|thanks|thank you|ok|okay|yes|no|cool|nice|good (morning|evening|night))\b[\s!.,?]*$", text):
            scores["general"] += 4.0

        if re.search(r"\b(plan|planning|roadmap|strategy|step[- ]by[- ]step|outline|milestones?|schedule|break (?:it|this) down|approach|walk me through|how should i (?:start|begin|build|structure))\b", text):
            scores["planning"] += 4.5
        if re.search(r"\b(debug|fix|bug|error|traceback|exception|code|python|js|java|function|class|api|refactor|unit test|syntax)\b", text):
            scores["coding"] += 4.0
        if re.search(r"\b(compare|tradeoff|analysis|analyze|reason|decision|design|architecture|optimi[sz]e|why|how to choose|pros|cons)\b", text):
            scores["reasoning"] += 4.0
        if re.search(r"\b(summarize|explain|what is|who is|overview|tell me about|plain english|in simple terms)\b", text):
            scores["general"] += 3.0
        if re.search(r"\b(image|picture|photo|chart|diagram|vision|describe the screenshot|what is in this)\b", text):
            scores["vision"] += 4.0
        if re.search(r"\b(audio|voice|speech|transcribe|listen|whisper|caption|recording|podcast)\b", text):
            scores["speech"] += 4.0
        if re.search(r"\b(organi[sz]e|sort|tidy|arrange|declutter|categori[sz]e|clean|duplicate|duplicates|large files|disk usage|system|folder|directory|clutter|clean up|empty folders?|undo)\b", text):
            scores["system_agent"] += 4.5
        # A named location on this PC is a strong signal the user means a real
        # folder ("analyze my desktop"), not a topic to talk about.
        if re.search(r"\b(desktop|downloads|my pc|this pc|hard drive|c drive)\b", text):
            scores["system_agent"] += 3.0
        if re.search(r"\b(should i|which is better|what would you recommend|based on the tradeoff)\b", text):
            scores["reasoning"] += 2.0
        if re.search(r"\b(project|document|readme|folder|source|knowledge base|according to|from my notes|local docs|local files|context)\b", text):
            scores["general"] += 1.5
        return scores

    def _semantic_scores(self, query: str) -> dict[str, float]:
        # Encode the query once; score against the pre-embedded exemplars.
        query_embedding = self.embed_model.encode([query], normalize_embeddings=True)[0]
        scores: dict[str, float] = {}
        for label, matrix in self.category_vectors.items():
            sims = matrix @ query_embedding
            scores[label] = float(sims.max()) if len(sims) else 0.0
        return scores

    def classify(self, query: str) -> dict[str, Any]:
        text = (query or "").strip()
        if not text:
            return {
                "task": "general",
                "confidence": 0.0,
                "scores": {name: 0.0 for name in self.TASKS},
            }

        rule_scores = self._rule_scores(text)
        semantic_scores = self._semantic_scores(text)
        # Increase weight of semantic scoring to prevent regex misclassifications on multi-intent prompts
        combined = {
            name: float(rule_scores.get(name, 0.0) + semantic_scores.get(name, 0.0) * 4.0)
            for name in self.TASKS
        }

        task = max(combined, key=combined.get)
        confidence = max(combined.values())
        return {"task": task, "confidence": confidence, "scores": combined}

    def _needs_rag(self, query: str) -> bool:
        text = query.lower()
        rag_signals = [
            "project",
            "document",
            "readme",
            "source",
            "according to",
            "local docs",
            "knowledge base",
            "my notes",
            "context",
            "file",
            "folder",
            "repo",
            "nexus",
        ]
        return any(signal in text for signal in rag_signals)

    def route(
        self,
        query: str,
        available_models: list[str] | None = None,
        gemini_ready: bool | None = None,
    ) -> dict[str, Any]:
        """Classify the query and pick the models to try, in order.

        Returns `model` (the first choice) plus `chain` — every reachable
        alternative, best first — so the engine can fall through without
        re-routing. Nothing here is user-configurable by design.
        """
        classification = self.classify(query)
        task = classification["task"]
        needs_rag = self._needs_rag(query)

        if task == "system_agent":
            chain: list[dict[str, Any]] = [
                {"model": "pc-toolkit", "provider": "toolkit", "score": 1.0,
                 "reason": "local file operation — no model needed"}
            ]
            complexity = 0.0
            avail = {"ollama": available_models or [], "gemini": bool(gemini_ready)}
        else:
            complexity = providers.estimate_complexity(query)
            avail = providers.availability(available_models, gemini_ready)
            chain = [
                {"model": c.model, "provider": c.spec.provider,
                 "score": round(c.score, 3), "reason": c.reason}
                for c in providers.plan(task, query, needs_rag, complexity, avail)
            ]

        top = chain[0] if chain else None
        decision = {
            "task": task,
            "model": top["model"] if top else None,
            "provider": top["provider"] if top else None,
            "chain": chain,
            "complexity": round(complexity, 2),
            "available": avail,
            "needs_rag": needs_rag,
            "confidence": classification["confidence"],
            "scores": classification["scores"],
            "reason": (
                f"Matched {task} intent using keyword and embedding cues; "
                + (top["reason"] if top else "no model reachable")
            ),
        }
        self.logger.log(decision)
        return decision


if __name__ == "__main__":
    router = TaskRouter()
    while True:
        prompt = input("NEXUS router> ")
        if not prompt:
            continue
        print(router.route(prompt))
