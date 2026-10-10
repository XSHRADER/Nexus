"""
router.py
Task router for NEXUS AI: classifies a query as general chat, coding,
reasoning, planning, or a PC action, decides when local document retrieval
should be included, and hands the task to `providers.plan()`, which picks the
model automatically.

Two classifiers, in this order:
  * the trained router (learned_router.py), when one has been trained and
    passed its gate, and it is confident enough;
  * keyword rules plus nearest labelled examples (router_examples.json),
    which always work and are what the trained router has to beat.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from nexus import config, providers
from nexus.embeddings import get_sentence_transformer

log = logging.getLogger(__name__)

LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUPS = 3
# How much of each question the routing log keeps. The log stays on this PC
# (data/router_logs.jsonl); the text is what lets a misrouted question be
# found later and turned into a labelled example.
LOG_QUERY_CHARS = 500

EXAMPLES_FILE = Path(__file__).with_name("router_examples.json")


def load_examples(path: str | Path = EXAMPLES_FILE) -> dict[str, list[str]]:
    """Read the labelled exemplar prompts, one list per task."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["examples"]


def preference_bonuses(task: str) -> dict[str, float]:
    """Score adjustments from your Arena results for this task.

    Only ratings that have settled (enough decided votes) count, scaled so
    200 Elo points above or below the starting 1000 is the full
    `preference_weight` -- enough to reorder near-equal models, not enough to
    hand a coding task to a chat model.
    """
    weight = config.get_settings().preference_weight
    if weight <= 0:
        return {}
    try:
        from nexus import feedback

        rows = feedback.leaderboard(task)
    except Exception:
        return {}
    return {
        r["model"]: weight * max(-1.0, min(1.0, (r["elo"] - feedback.ELO_START) / 200.0))
        for r in rows if r.get("settled")
    }


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
    """Rules and labelled examples, with the trained router in front when it exists."""

    TASKS = (*providers.TEXT_TASKS, "system_agent")
    # Not something a text prompt is classified as: set when an image is attached.
    FORCED_ONLY_TASKS = ("vision",)

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

    # Labelled prompts the semantic half of the classifier compares against.
    # Kept as data so new phrasings can be taught without touching code.
    CATEGORY_EXAMPLES = load_examples()
    # A task's semantic score is the mean similarity of its TOP_K nearest
    # exemplars (a single best match let one lucky phrasing decide), scaled by
    # SEMANTIC_WEIGHT against the keyword rules. Both were picked by
    # leave-one-out accuracy over the exemplars, not on eval/router_set.json.
    TOP_K = 6
    SEMANTIC_WEIGHT = 8.0

    def __init__(self, embedding_model: str = config.EMBED_MODEL, use_learned: bool = True):
        self.embed_model = get_sentence_transformer(embedding_model)
        self.logger = DecisionLogger()
        # The trained router, when one has been trained and passed its gate.
        # Every decision it makes has the rules below as a fallback: low
        # confidence, a missing head, or `[router] mode = "rules"`.
        self.use_learned = use_learned
        self.learned = None
        self.reload_learned()
        # Pre-embed the category exemplars once; reused on every query.
        self.category_vectors = {
            label: self.embed_model.encode(examples, normalize_embeddings=True)
            for label, examples in self.CATEGORY_EXAMPLES.items()
        }

    def _rule_scores(self, query: str) -> dict[str, float]:
        text = query.lower()
        scores = {name: 0.0 for name in self.TASKS}
        # No standing head start for chat any more: with a full exemplar set
        # it mostly dragged real coding/planning prompts into "general".
        # Greetings still get a strong explicit push here.
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
            top = np.sort(sims)[-self.TOP_K:]
            scores[label] = float(top.mean()) if len(top) else 0.0
        return scores

    def reload_learned(self) -> None:
        """Pick up a newly trained router (or drop it) without restarting."""
        self.learned = None
        if not self.use_learned or config.get_settings().router_mode == "rules":
            return
        try:
            from nexus import learned_router

            self.learned = learned_router.load_current()
        except Exception as exc:
            log.warning("could not load the trained router: %s", exc)
            self.learned = None

    def _learned(self, query: str) -> dict[str, dict[str, float]] | None:
        if self.learned is None:
            return None
        try:
            return self.learned.predict(query)
        except Exception:
            return None

    def classify(self, query: str) -> dict[str, Any]:
        text = (query or "").strip()
        if not text:
            return {
                "task": "general",
                "confidence": 0.0,
                "scores": {name: 0.0 for name in self.TASKS},
                "method": "rules",
            }

        settings = config.get_settings()
        # The trained task head also knows "vision" and "speech"; a text
        # prompt is never routed to those, so only the text tasks compete.
        learned = (self._learned(text) or {}).get("task") or {}
        probs = {name: float(learned.get(name, 0.0)) for name in self.TASKS} if learned else {}
        if probs:
            task = max(probs, key=probs.get)
            if probs[task] >= settings.router_min_confidence or settings.router_mode == "learned":
                return {"task": task, "confidence": probs[task], "scores": probs,
                        "method": f"learned {self.learned.version}"}
        result = self._rule_classify(text)
        result["method"] = "rules (learned router unsure)" if probs else "rules"
        return result

    def _rule_classify(self, text: str) -> dict[str, Any]:
        rule_scores = self._rule_scores(text)
        semantic_scores = self._semantic_scores(text)
        combined = {
            name: float(rule_scores.get(name, 0.0) + semantic_scores.get(name, 0.0) * self.SEMANTIC_WEIGHT)
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
        """Does answering this need your documents? Learned when available."""
        probs = (self._learned(query) or {}).get("docs")
        if probs:
            return probs.get("yes", 0.0) >= config.get_settings().docs_threshold
        return self._rule_needs_rag(query)

    def _rule_needs_rag(self, query: str) -> bool:
        return bool(self._RAG_SIGNALS.search(query.lower()))

    def needs_strong(self, query: str) -> tuple[bool | None, float | None]:
        """(is this a hard prompt for cloud purposes?, P(strong)) -- None when
        no strong head is trained, which leaves the difficulty estimate in charge."""
        probs = (self._learned(query) or {}).get("strong")
        if not probs:
            return None, None
        p = probs.get("strong", 0.0)
        return p >= config.get_settings().strong_threshold, p

    def route(
        self,
        query: str,
        available_models: list[str] | None = None,
        force_task: str | None = None,
        policy: Any = None,
        needs_image: bool = False,
        rag_override: bool | None = None,
    ) -> dict[str, Any]:
        """Classify the query and pick the models to try, in order.

        Returns `model` (the first choice) plus `chain` — every reachable
        alternative, best first — so the engine can fall through without
        re-routing. `force_task` overrides the classifier.

        `policy` (cloud.CloudPolicy) says whether cloud models may compete;
        `needs_image` keeps only models that read images; `rag_override`
        replaces the guess about using your documents, which matters here
        because a document question may not be allowed to leave this PC.
        """
        classification = self.classify(query)
        task = classification["task"]
        # An explicit override still records what the classifier *would* have
        # said, so the UI can show that the two disagreed.
        if force_task and force_task in (*self.TASKS, *self.FORCED_ONLY_TASKS):
            classification = dict(classification, auto_task=task, forced=True)
            task = force_task
        needs_rag = self._needs_rag(query) if rag_override is None else bool(rag_override)
        hard, p_strong = self.needs_strong(query)

        if task == "system_agent":
            chain: list[dict[str, Any]] = [
                {"model": "pc-toolkit", "provider": "toolkit", "local": True, "score": 1.0,
                 "reason": "local file operation — no model needed"}
            ]
            complexity = 0.0
            avail = {"ollama": list(available_models or []), "ollama_up": None, "loaded": []}
        else:
            complexity = providers.estimate_complexity(query)
            avail = providers.availability(available_models)
            chain = [
                {"model": c.model, "provider": c.spec.provider, "local": c.spec.is_local,
                 "score": round(c.score, 3), "reason": c.reason}
                for c in providers.plan(
                    task, query, complexity, avail, needs_rag=needs_rag, policy=policy,
                    needs_image=needs_image, hard=hard, bonuses=preference_bonuses(task),
                )
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
            "router_method": classification.get("method", "rules"),
            "p_strong": None if p_strong is None else round(p_strong, 3),
            "reason": (
                (f"Classified as {task} by the {classification['method']} router "
                 f"({classification['confidence']:.0%} sure); "
                 if str(classification.get("method", "")).startswith("learned")
                 else f"Matched {task} intent using keyword and embedding cues; ")
                + (top["reason"] if top else "no model reachable")
            ),
        }
        self.logger.log(dict(decision, query=query[:LOG_QUERY_CHARS]))
        return decision


if __name__ == "__main__":
    router = TaskRouter()
    while True:
        prompt = input("NEXUS router> ")
        if not prompt:
            continue
        print(router.route(prompt))
