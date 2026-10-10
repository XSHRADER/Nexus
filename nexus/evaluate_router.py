"""
evaluate_router.py
Scores the task router against a held-out set of labelled prompts, so changes
to its rules or exemplars can be judged instead of guessed at.

The prompts in eval/router_set.json are deliberately *not* in the router's own
exemplar file (nexus/router_examples.json). Copying one across would make the
score look better without the router getting any better.

Usage
    python -m nexus.evaluate_router            # accuracy, per task, misses
    python -m nexus.evaluate_router --quiet    # summary only

The first block scores the rules and examples alone. When a learned router
has been trained, a second line scores it on the same prompts.
"""

import argparse
import json
from collections import Counter

from nexus import config
from nexus.router import TaskRouter

ROUTER_SET = config.PROJECT_DIR / "eval" / "router_set.json"


def load_set() -> list[dict]:
    with open(ROUTER_SET, encoding="utf-8") as f:
        return json.load(f)["queries"]


def evaluate(router: TaskRouter, queries: list[dict]) -> dict:
    """Classify every query; return accuracy, per-task recall and the misses."""
    totals: Counter = Counter()
    correct: Counter = Counter()
    confusion: Counter = Counter()
    misses = []
    for item in queries:
        expected = item["task"]
        got = router.classify(item["q"])["task"]
        totals[expected] += 1
        confusion[(expected, got)] += 1
        if got == expected:
            correct[expected] += 1
        else:
            misses.append((item["q"], expected, got))
    n = sum(totals.values())
    return {
        "accuracy": sum(correct.values()) / n if n else 0.0,
        "n": n,
        "per_task": {t: (correct[t], totals[t]) for t in totals},
        "confusion": confusion,
        "misses": misses,
    }


def accuracy(classify, queries: list[dict] | None = None) -> float:
    """Share of the held-out prompts `classify(prompt) -> task` gets right."""
    queries = queries if queries is not None else load_set()
    return sum(classify(q["q"]) == q["task"] for q in queries) / max(len(queries), 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate NEXUS task routing.")
    parser.add_argument("--quiet", action="store_true", help="Print the summary only.")
    args = parser.parse_args()

    router = TaskRouter(use_learned=False)  # the rules and examples, not a trained router
    result = evaluate(router, load_set())

    exemplars = sum(len(v) for v in router.CATEGORY_EXAMPLES.values())
    print(f"Router accuracy: {result['accuracy']:.1%} on {result['n']} held-out prompts "
          f"({exemplars} exemplars loaded)\n")
    print(f"{'task':<14}{'correct':>10}{'recall':>9}")
    for task in router.TASKS:
        ok, total = result["per_task"].get(task, (0, 0))
        recall = f"{ok / total:.0%}" if total else "-"
        print(f"{task:<14}{ok:>6}/{total:<3}{recall:>9}")

    if not args.quiet and result["misses"]:
        print(f"\nMisses ({len(result['misses'])}):")
        for q, expected, got in result["misses"]:
            print(f"  [{expected} -> {got}] {q}")

    from nexus import learned_router
    from nexus.evaluate_learned_router import learned_functions

    current = learned_router.load_current()
    if current is not None and "task" in current.heads:
        classify, _docs = learned_functions(current)
        print(f"\nLearned router {current.version} on the same prompts: {accuracy(classify):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
