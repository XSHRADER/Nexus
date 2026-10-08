"""
eval_truth.py
Measures the truth check against labelled claims, so "the green ticks mean
something" is a number rather than a promise.

Two measurements, each for every available checker (keyword baseline, NLI):

  given passage   the checker sees the exact passage each claim was written
                  against. This isolates the judging step: can it tell
                  "says the same", "says something else" and "doesn't say"
                  apart?
  end to end      the checker has to find its own evidence among all the
                  passages in documents/, as it does inside NEXUS. Lower is
                  expected: a wrong passage can turn a supported claim into
                  "not found".

Reported: accuracy, macro-F1, per-label precision/recall, and the confusion
matrix (rows = true label, columns = predicted).

Usage
    python eval_truth.py                 # every checker that can load
    python eval_truth.py --method keyword
    python eval_truth.py --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import truth_check as tc
from config import get_settings

PROJECT_DIR = Path(__file__).resolve().parent
GOLDEN = PROJECT_DIR / "eval" / "claims_golden.json"


def load_items(path: Path = GOLDEN) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["claims"]


def metrics(truth: list[str], pred: list[str]) -> dict:
    labels = list(tc.LABELS)
    confusion = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(truth, pred):
        confusion[t][p] += 1
    per_label = {}
    for label in labels:
        tp = confusion[label][label]
        predicted = sum(confusion[t][label] for t in labels)
        actual = sum(confusion[label].values())
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"precision": round(precision, 3), "recall": round(recall, 3),
                            "f1": round(f1, 3)}
    accuracy = sum(t == p for t, p in zip(truth, pred)) / len(truth) if truth else 0.0
    return {
        "n": len(truth),
        "accuracy": round(accuracy, 3),
        "macro_f1": round(sum(v["f1"] for v in per_label.values()) / len(labels), 3),
        "per_label": per_label,
        "confusion": confusion,
    }


def evaluate_given(scorer, items: list[dict]) -> dict:
    s = get_settings()
    probs = scorer.score([(it["premise"], it["claim"]) for it in items])
    pred = [tc.label_for(p, s.support_threshold, s.contradict_threshold)[0] for p in probs]
    out = metrics([it["label"] for it in items], pred)
    out["errors"] = [
        {"id": it["id"], "claim": it["claim"], "true": it["label"], "predicted": p}
        for it, p in zip(items, pred) if p != it["label"]
    ]
    return out


def evaluate_end_to_end(scorer, items: list[dict]) -> dict:
    passages = tc.document_passages()
    pred = []
    for it in items:
        report = tc.check(it["claim"], scorer=scorer, passages=passages)
        claims = report["claims"]
        pred.append(claims[0]["label"] if claims else tc.NOT_FOUND)
    return metrics([it["label"] for it in items], pred)


def available_scorers(method: str | None = None) -> list:
    scorers = []
    if method in (None, "all", "keyword"):
        scorers.append(tc.KeywordScorer())
    if method in (None, "all", "nli"):
        try:
            scorers.append(tc.NLIScorer(get_settings().truth_model))
        except Exception as exc:  # model or package unavailable
            if method == "nli":
                raise
            print(f"(NLI checker unavailable: {exc.__class__.__name__}: {exc})")
    return scorers


def run(method: str | None = None) -> dict:
    items = load_items()
    results = {}
    for scorer in available_scorers(method):
        results[scorer.name] = {
            "given_passage": evaluate_given(scorer, items),
            "end_to_end": evaluate_end_to_end(scorer, items),
        }
    return results


def print_table(results: dict) -> None:
    print(f"\nTruth check on {len(load_items())} labelled claims "
          "(20 supported / 20 contradicted / 20 not found)\n")
    print(f"{'checker':<38} {'mode':<14} {'accuracy':>8} {'macro-F1':>9} "
          f"{'contra. recall':>15}")
    for name, modes in results.items():
        for mode, m in modes.items():
            print(f"{name:<38} {mode:<14} {m['accuracy']:>8.3f} {m['macro_f1']:>9.3f} "
                  f"{m['per_label']['contradicted']['recall']:>15.3f}")
    for name, modes in results.items():
        m = modes["given_passage"]
        print(f"\nConfusion, {name}, given passage (rows = true, cols = predicted):")
        print(f"{'':<14}" + "".join(f"{p:>14}" for p in tc.LABELS))
        for t in tc.LABELS:
            print(f"{t:<14}" + "".join(f"{m['confusion'][t][p]:>14}" for p in tc.LABELS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--method", choices=["all", "keyword", "nli"], default="all")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    results = run(args.method)
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_table(results)
