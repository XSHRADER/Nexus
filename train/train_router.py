"""
train/train_router.py
Trains the learned router, measures it, and switches to it only if it wins.

    python train/train_router.py                 # seed + teacher + your feedback
    python train/train_router.py --arena55k      # also learn "needs a strong model"
                                                 # from 55k public votes (needs
                                                 # `pip install datasets`)
    python run.py --train                        # the same as the first line

Steps
  1. Gather rows for each head (train/data_sources.py).
  2. Train the task and documents heads on all their rows; train the strong
     head on 80% of its rows and keep 20% back to measure it.
  3. Measure task and documents decisions on eval/router_golden.json (never
     trained on), for the hand-written rules, the router currently in use,
     and the new one.
  4. Gate: the new router becomes current only if it beats the rules on task
     accuracy and matches or beats them on the documents F1, and doesn't fall
     below the current router on either. A strong head that can't beat
     random routing on its held-out votes is left out.
  5. Save models/router_vN.npz + .json (metrics included); move
     models/current.json only if the gate passed (or with --force).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import eval_router  # noqa: E402
import learned_router as lr  # noqa: E402
from train import data_sources  # noqa: E402

MIN_STRONG_ROWS = 50


# ---------------------------------------------------------------------------
# Strong head evaluation (RouteLLM-style)
# ---------------------------------------------------------------------------


def auc(labels: list[int], scores: list[float]) -> float:
    """Area under the ROC curve: P(a random strong-needed prompt scores above
    a random weak-enough one). 0.5 is chance."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def quality_curve(outcomes: list[str], p_strong: list[float]) -> dict:
    """Send the top fraction f of prompts (by P(strong)) to the strong model.
    Quality = share of prompts where the routed model won or tied. Compared
    with sending a random fraction f, which is a straight line between
    all-weak and all-strong."""
    outcomes = np.asarray(outcomes)
    order = np.argsort(-np.asarray(p_strong), kind="mergesort")
    n = len(outcomes)
    ok_strong = np.isin(outcomes, ["strong", "tie"])
    ok_weak = np.isin(outcomes, ["weak", "tie"])
    q_strong, q_weak = float(ok_strong.mean()), float(ok_weak.mean())
    points = []
    for f in [i / 10 for i in range(11)]:
        k = int(round(f * n))
        routed_strong = np.zeros(n, bool)
        routed_strong[order[:k]] = True
        q = float(np.where(routed_strong, ok_strong, ok_weak).mean())
        points.append({"strong_calls": f, "quality": round(q, 4),
                       "random": round(f * q_strong + (1 - f) * q_weak, 4)})
    target = q_weak + 0.95 * (q_strong - q_weak)
    cpt95 = next((p["strong_calls"] for p in points if p["quality"] >= target), 1.0)
    gain = float(np.mean([p["quality"] - p["random"] for p in points]))
    return {"all_weak": round(q_weak, 4), "all_strong": round(q_strong, 4), "points": points,
            "strong_calls_for_95pct": cpt95, "mean_gain_over_random": round(gain, 4)}


# ---------------------------------------------------------------------------


def train_head(rows: list[dict], classes: list[str], use_embeddings: bool, **fit_args):
    X = lr.featurize([r["prompt"] for r in rows], use_embeddings=use_embeddings)
    model = lr.LinearModel(classes, X.n_features, use_embeddings)
    return model.fit(X, [r["label"] for r in rows], [r["weight"] for r in rows], **fit_args)


def cross_validate(rows: list[dict], classes: list[str], use_embeddings: bool, folds: int = 5) -> float:
    """Accuracy over k folds of the training rows -- a check that the model
    generalises within its own data, separate from the golden set."""
    rng = np.random.default_rng(0)
    order = rng.permutation(len(rows))
    hits = 0
    for k in range(folds):
        test_idx = set(order[k::folds].tolist())
        train = [r for i, r in enumerate(rows) if i not in test_idx]
        test = [rows[i] for i in sorted(test_idx)]
        model = train_head(train, classes, use_embeddings)
        pred = model.predict(lr.featurize([r["prompt"] for r in test], use_embeddings))
        hits += sum(p == r["label"] for p, r in zip(pred, test))
    return round(hits / len(rows), 3)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--embeddings", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--arena55k", action="store_true",
                        help="download the public votes and train the strong head")
    parser.add_argument("--limit", type=int, default=None, help="use at most N public battles")
    parser.add_argument("--no-own", action="store_true", help="ignore your own feedback")
    parser.add_argument("--force", action="store_true", help="make current even if the gate fails")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    started = time.time()

    arena_info = None
    if args.arena55k:
        arena_info = data_sources.prepare_arena55k(args.limit)
        print(f"arena55k: {arena_info['battles_used']} strong-vs-weak battles from "
              f"{arena_info['battles_total']}; licence on the dataset card: "
              f"{arena_info.get('license') or 'not stated'}")
        print(f"  strong tier: {', '.join(arena_info['strong_models'])}")

    use_emb = {"on": True, "off": False}.get(args.embeddings)
    if use_emb is None:
        use_emb = lr.embeddings_available()

    task_rows, docs_rows = data_sources.task_and_docs_rows(include_own=not args.no_own)
    strong_rows = data_sources.strong_head_rows(include_own=not args.no_own)
    tasks = list(data_sources.TASKS)

    heads = {
        "task": train_head(task_rows, tasks, use_emb),
        "docs": train_head(docs_rows, ["no", "yes"], use_emb),
    }
    meta = {
        "version": lr.next_version(),
        "created": time.time(),
        "feature_version": lr.FEATURE_VERSION,
        "use_embeddings": use_emb,
        "heads": {
            "task": {"classes": tasks, "rows": data_sources.counts(task_rows)},
            "docs": {"classes": ["no", "yes"], "rows": data_sources.counts(docs_rows)},
        },
        "metrics": {"task_cv_accuracy": cross_validate(task_rows, tasks, use_emb)},
    }

    # Strong head: train on 80%, measure on the 20% it never saw.
    strong_report = None
    labels = {r["label"] for r in strong_rows}
    if len(strong_rows) >= MIN_STRONG_ROWS and labels == {"strong", "weak"}:
        rng = np.random.default_rng(0)
        order = rng.permutation(len(strong_rows))
        cut = int(len(order) * 0.8)
        train = [strong_rows[i] for i in order[:cut]]
        test = [strong_rows[i] for i in order[cut:]]
        head = train_head(train, ["weak", "strong"], use_emb,
                          epochs=8 if len(train) > 5000 else 30, batch_size=256)
        probs = head.predict_proba(lr.featurize([r["prompt"] for r in test], use_emb))[:, 1]
        strong_report = {
            "auc": round(auc([r["label"] == "strong" for r in test], probs.tolist()), 4),
            "held_out": len(test),
            **quality_curve([r.get("outcome", r["label"]) for r in test], probs.tolist()),
        }
        strong_report["passed"] = bool(strong_report["auc"] >= 0.55
                                       and strong_report["mean_gain_over_random"] > 0)
        meta["metrics"]["strong"] = strong_report
        if strong_report["passed"]:
            heads["strong"] = head
            meta["heads"]["strong"] = {"classes": ["weak", "strong"],
                                       "rows": data_sources.counts(strong_rows)}
    if arena_info:
        meta["arena55k"] = arena_info

    new_router = lr.LearnedRouter(heads, meta)
    golden = eval_router.load_golden()
    rules = eval_router.evaluate(*eval_router.rules_functions(), golden)
    new = eval_router.evaluate(*eval_router.learned_functions(new_router), golden)
    previous = lr.load_current()
    prev = (eval_router.evaluate(*eval_router.learned_functions(previous), golden)
            if previous is not None and {"task", "docs"} <= set(previous.heads) else None)

    gate = {
        "task_beats_rules": new["task"]["accuracy"] > rules["task"]["accuracy"],
        "docs_not_worse_than_rules": new["docs"]["f1"] >= rules["docs"]["f1"],
        "task_not_worse_than_current": prev is None or new["task"]["accuracy"] >= prev["task"]["accuracy"],
        "docs_not_worse_than_current": prev is None or new["docs"]["f1"] >= prev["docs"]["f1"],
    }
    passed = all(gate.values())
    meta["metrics"]["golden"] = {"rules": _brief(rules), "new": _brief(new),
                                 "previous": _brief(prev) if prev else None}
    meta["gate"] = gate
    meta["passed_gate"] = passed
    meta["train_seconds"] = round(time.time() - started, 1)
    lr.save_version(new_router, make_current=passed or args.force)

    report = {"version": meta["version"], "made_current": passed or args.force, **meta}
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_report(report, previous.version if previous else None)
    return report


def _brief(r: dict) -> dict:
    return {"task_accuracy": r["task"]["accuracy"], "task_macro_f1": r["task"]["macro_f1"],
            "docs_precision": r["docs"]["precision"], "docs_recall": r["docs"]["recall"],
            "docs_f1": r["docs"]["f1"], "docs_false_alarms": r["docs"]["false_pos"]}


def print_report(report: dict, previous_version: str | None) -> None:
    m = report["metrics"]
    print(f"\nRouter {report['version']} (features: hashed n-grams"
          f"{' + MiniLM embeddings' if report['use_embeddings'] else ''}) "
          f"trained in {report['train_seconds']}s")
    for head in ("task", "docs", "strong"):
        if head in report["heads"]:
            print(f"  {head:<6} rows: {report['heads'][head]['rows']}")
    print(f"  task cross-validation accuracy on its training rows: {m['task_cv_accuracy']}")
    g = m["golden"]
    print("\nHeld-out routing set (eval/router_golden.json):")
    print(f"  {'router':<14} {'task acc':>9} {'task F1':>8} {'docs P':>7} {'docs R':>7} "
          f"{'docs F1':>8} {'false alarms':>13}")
    rows = [("rules", g["rules"]), (f"previous {previous_version}", g["previous"]),
            (f"new {report['version']}", g["new"])]
    for name, r in rows:
        if r:
            print(f"  {name:<14} {r['task_accuracy']:>9.3f} {r['task_macro_f1']:>8.3f} "
                  f"{r['docs_precision']:>7.3f} {r['docs_recall']:>7.3f} {r['docs_f1']:>8.3f} "
                  f"{r['docs_false_alarms']:>13}")
    s = m.get("strong")
    if s:
        print(f"\nStrong-vs-weak head on {s['held_out']} held-out votes: AUC {s['auc']}, "
              f"{'kept' if s['passed'] else 'left out (no better than random)'}")
        print(f"  all-weak quality {s['all_weak']}, all-strong {s['all_strong']}; "
              f"strong calls for 95% of the gap: {s['strong_calls_for_95pct']:.0%}")
        print("  strong calls  quality  random")
        for p in s["points"]:
            print(f"  {p['strong_calls']:>11.0%}  {p['quality']:>7.3f}  {p['random']:>6.3f}")
    print("\nGate: " + ", ".join(f"{k}={'yes' if v else 'NO'}" for k, v in report["gate"].items()))
    print(f"-> {report['version']} {'is now in use' if report['made_current'] else 'saved but NOT in use'}")


if __name__ == "__main__":
    main()
