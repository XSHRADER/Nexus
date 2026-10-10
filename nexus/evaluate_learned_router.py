"""
evaluate_learned_router.py
Scores routing on the held-out prompts in eval/router_golden.json: the rules
and labelled examples against the learned router, on both decisions the
router makes before any model runs.

  task       accuracy and macro-F1 over the tasks a text prompt can be routed
             to, plus the confusion matrix. The file also holds "vision" and
             "speech" prompts; those are left out, because a text prompt is
             never routed there (a vision model is only used when an image is
             attached), so neither router could be right on them.
  documents  precision / recall / F1 of "this needs your documents"; reported
             this way because only ~10% of prompts need documents, so a gate
             that always says "no" would score 90% accuracy

The golden set is never used for training or for choosing settings; it is
the gate train/train_router.py uses to decide whether a new model replaces
the old one.

Usage
    python -m nexus.evaluate_learned_router           # rules vs the current learned router
    python -m nexus.evaluate_learned_router --json

`python -m nexus.evaluate_router` scores the rules alone on a larger set.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nexus import learned_router
from nexus.config import PROJECT_DIR
from nexus.router import TaskRouter

GOLDEN = PROJECT_DIR / "eval" / "router_golden.json"
TASKS = TaskRouter.TASKS


def load_golden(path: Path = GOLDEN) -> list[dict[str, Any]]:
    """The held-out prompts whose task a text prompt can actually be routed to."""
    prompts = json.loads(path.read_text(encoding="utf-8"))["prompts"]
    return [p for p in prompts if p["task"] in TASKS]


def task_metrics(truth: list[str], pred: list[str]) -> dict[str, Any]:
    confusion = {t: {p: 0 for p in TASKS} for t in TASKS}
    for t, p in zip(truth, pred):
        confusion[t][p] += 1
    f1s = []
    for label in TASKS:
        tp = confusion[label][label]
        predicted = sum(confusion[t][label] for t in TASKS)
        actual = sum(confusion[label].values())
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "accuracy": round(sum(t == p for t, p in zip(truth, pred)) / max(len(truth), 1), 3),
        "macro_f1": round(sum(f1s) / len(f1s), 3),
        "confusion": confusion,
    }


def binary_metrics(truth: list[bool], pred: list[bool]) -> dict[str, Any]:
    tp = sum(t and p for t, p in zip(truth, pred))
    fp = sum(p and not t for t, p in zip(truth, pred))
    fn = sum(t and not p for t, p in zip(truth, pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
            "true_pos": tp, "false_pos": fp, "false_neg": fn,
            "accuracy": round(sum(t == p for t, p in zip(truth, pred)) / max(len(truth), 1), 3)}


def evaluate(classify: Callable[[str], str], needs_docs: Callable[[str], bool],
             golden: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    golden = golden or load_golden()
    pred_task = [classify(g["prompt"]) for g in golden]
    pred_docs = [bool(needs_docs(g["prompt"])) for g in golden]
    out = {
        "task": task_metrics([g["task"] for g in golden], pred_task),
        "docs": binary_metrics([g["needs_docs"] for g in golden], pred_docs),
    }
    out["task"]["errors"] = [
        {"id": g["id"], "prompt": g["prompt"], "true": g["task"], "predicted": p}
        for g, p in zip(golden, pred_task) if p != g["task"]
    ]
    return out


def rules_functions():
    router = TaskRouter(use_learned=False)
    return (lambda q: router.classify(q)["task"]), router._rule_needs_rag


def learned_functions(model: learned_router.LearnedRouter, threshold: float = 0.5):
    def classify(q: str) -> str:
        # As in TaskRouter.classify: only the text tasks compete.
        probs = model.predict(q)["task"]
        return max(TASKS, key=lambda t: probs.get(t, 0.0))

    def docs(q: str) -> bool:
        return model.predict(q)["docs"].get("yes", 0.0) >= threshold

    return classify, docs


def run() -> dict[str, Any]:
    results = {"rules": evaluate(*rules_functions())}
    current = learned_router.load_current()
    if current is not None and {"task", "docs"} <= set(current.heads):
        results[f"learned {current.version}"] = evaluate(*learned_functions(current))
    return results


def print_table(results: dict[str, Any]) -> None:
    n = len(load_golden())
    print(f"\nRouting on {n} held-out prompts ({n // len(TASKS)} per task; "
          f"{sum(g['needs_docs'] for g in load_golden())} need documents)\n")
    print(f"{'router':<22} {'task acc':>9} {'task F1':>8}   {'docs P':>7} {'docs R':>7} "
          f"{'docs F1':>8} {'false alarms':>13}")
    for name, r in results.items():
        t, d = r["task"], r["docs"]
        print(f"{name:<22} {t['accuracy']:>9.3f} {t['macro_f1']:>8.3f}   {d['precision']:>7.3f} "
              f"{d['recall']:>7.3f} {d['f1']:>8.3f} {d['false_pos']:>13}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    out = run()
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print_table(out)
        for name, r in out.items():
            if r["task"]["errors"]:
                print(f"\n{name} task errors:")
                for e in r["task"]["errors"][:15]:
                    print(f"  {e['true']:>12} -> {e['predicted']:<12} {e['prompt']}")
