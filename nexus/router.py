"""
router.py
Task router for NEXUS AI: classifies a query as general chat, coding,
reasoning, planning, or a PC action, decides when local document retrieval
should be included, and hands the task to `providers.plan()`, which picks the
model automatically.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

from nexus import config, providers
from nexus.embeddings import get_sentence_transformer

log = logging.getLogger(__name__)

LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUPS = 3


class DecisionLogger:
    """Appends routing decisions as JSON lines, rotating at `max_bytes`.

    router_logs.jsonl -> .1 -> .2 -> .3; the oldest falls off the end, so the
    log stays under roughly (backups + 1) * max_bytes.
    """

    def __init__(
        self,
        log_file: str | Path = config.ROUTER_LOG,
        max_bytes: int = LOG_MAX_BYTES,
        backups: int = LOG_BACKUPS,
    ):
        self.log_file = Path(log_file)
        self.max_bytes = max_bytes
        self.backups = backups
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def _backup(self, n: int) -> Path:
        return self.log_file.with_name(f"{self.log_file.name}.{n}")

    def _rotate(self) -> None:
        for n in range(self.backups, 0, -1):
            src = self.log_file if n == 1 else self._backup(n - 1)
            if src.exists():
                src.replace(self._backup(n))

    def log(self, payload: dict[str, Any]) -> None:
        try:
            if self.log_file.stat().st_size >= self.max_bytes:
                self._rotate()
        except OSError:
            # Missing file, or (on Windows) another process has it open —
            # skip rotating this time rather than lose the decision.
            pass
        with open(self.log_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


class TaskRouter:
    """Simple rule-based router with lightweight semantic fallback."""

    TASKS = (*providers.TEXT_TASKS, "system_agent")

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

    def __init__(self, embedding_model: str = config.EMBED_MODEL):
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
        # signal at all and embedding noise can hand it to any other task.
        scores["general"] += 1.2
        if re.match(r"^\s*(hi|hey|hello|yo|sup|thanks|thank you|ok|okay|yes|no|cool|nice|good (morning|evening|night))\b[\s!.,?]*$", text):
            scores["general"] += 4.0

        if re.search(r"\b(plan|planning|roadmap|strategy|step[- ]by[- ]step|outline|milestones?|schedule|break (?:it|this) down|approach|walk me through|how should i (?:start|begin|build|structure))\b", text):
            scores["planning"] += 4.5

        # A coding *request* needs an action or a failure, not merely the name
        # of a technology. "what is an API?" and "what Python version is used?"
        # were scoring full coding intent off the bare noun alone and going to
        # the coder model instead of being answered as ordinary questions.
        if re.search(
            r"\b(debug|refactor|implement|rewrite|fix (?:this|the|my|a)|"
            r"write (?:a |an )?(?:\w+ )?(?:function|class|script|test|query|method)|"
            r"unit test|stack ?trace|traceback|exception|typeerror|valueerror|"
            r"keyerror|attributeerror|indexerror|syntax error|compile|"
            r"runtime error)\b",
            text,
        ):
            scores["coding"] += 4.0
        if re.search(r"\b(bug|crash(?:es|ed|ing)?|broken|failing|fails)\b", text):
            scores["coding"] += 2.5
        # Technology nouns on their own are only weak evidence.
        if re.search(
            r"\b(code|python|javascript|js|java|typescript|function|class|api|"
            r"variable|regex|sql|error)\b",
            text,
        ):
            scores["coding"] += 1.5

        # `tradeoff\b` never matched the plural, which is how most people
        # actually write it.
        if re.search(r"\b(compare|tradeoffs?|trade[- ]offs?|analysis|analyze|reason|decision|design|architecture|optimi[sz]e|why|how to choose|pros|cons)\b", text):
            scores["reasoning"] += 4.0

        if re.search(
            r"\b(summarize|explain|what is|who is|overview|tell me about|"
            r"plain english|in simple terms|conceptually|high[- ]level)\b",
            text,
        ):
            scores["general"] += 3.0
        # Questions about versions and setup are informational, not coding work.
        if re.search(r"\b(version|release|installed|which model|what model)\b", text):
            scores["general"] += 2.5

        # Same principle as coding: a file operation needs a verb. Bare
        # "folder", "directory" or "system" was turning questions *about* the
        # project into requests to act on the disk.
        if re.search(
            r"\b(organi[sz]e|sort|tidy|arrange|declutter|categori[sz]e|"
            r"clean (?:up|out)|find duplicates?|duplicate files|large files|"
            r"disk usage|empty folders?|undo)\b",
            text,
        ):
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

    # Whole words only: as substrings, "file" matched "profile" and "source"
    # matched "resources", pulling documents into unrelated questions.
    _RAG_SIGNALS = re.compile(
        r"\b(project|documents?|docs|readme|sources?|according to|knowledge base|"
        r"my notes|notes|context|files?|folders?|repo|nexus)\b"
    )

    def _needs_rag(self, query: str) -> bool:
        return bool(self._RAG_SIGNALS.search(query.lower()))

    def route(
        self,
        query: str,
        available_models: list[str] | None = None,
        force_task: str | None = None,
    ) -> dict[str, Any]:
        """Classify the query and pick the models to try, in order.

        Returns `model` (the first choice) plus `chain` — every reachable
        alternative, best first — so the engine can fall through without
        re-routing. `force_task` overrides the classifier.
        """
        classification = self.classify(query)
        task = classification["task"]
        # An explicit override still records what the classifier *would* have
        # said, so the UI can show that the two disagreed.
        if force_task and force_task in self.TASKS:
            classification = dict(classification, auto_task=task, forced=True)
            task = force_task
        needs_rag = self._needs_rag(query)

        if task == "system_agent":
            chain: list[dict[str, Any]] = [
                {"model": "pc-toolkit", "provider": "toolkit", "score": 1.0,
                 "reason": "local file operation — no model needed"}
            ]
            complexity = 0.0
            avail = {"ollama": list(available_models or []), "ollama_up": None, "loaded": []}
        else:
            complexity = providers.estimate_complexity(query)
            avail = providers.availability(available_models)
            chain = [
                {"model": c.model, "provider": c.spec.provider,
                 "score": round(c.score, 3), "reason": c.reason}
                for c in providers.plan(task, query, complexity, avail)
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
            "auto_task": classification.get("auto_task", task),
            "forced": bool(classification.get("forced")),
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
