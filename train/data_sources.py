"""
train/data_sources.py
Where the router's training examples come from. Every row is
{"prompt", "label", "weight", "source"}, one list per head.

  task / docs heads
    seed      train/data/seed_tasks.jsonl: hand-labelled starter prompts
    teacher   train/data/teacher_labels.jsonl: prompts labelled by a cloud
              model (train/teacher_label.py), e.g. public prompts
    yours     your corrections: task overrides and document on/off overrides
              (weighted 3x -- they are about *your* use, not the average)

  strong head
    arena55k  lmarena-ai/arena-human-preference-55k: 55k human votes between
              two chatbots. Models are split into a strong and a weak tier by
              their win rate *in the data*; a battle between the tiers says
              whether the prompt needed the strong one.
    yours     your Arena battles between a local and a cloud model
              (weighted 5x): did the cloud model win for this prompt?
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from config import PROJECT_DIR
from learned_router import iter_jsonl

DATA_DIR = PROJECT_DIR / "train" / "data"
SEED = DATA_DIR / "seed_tasks.jsonl"
TEACHER = DATA_DIR / "teacher_labels.jsonl"
ARENA55K = DATA_DIR / "arena55k_strong.jsonl"
ARENA55K_REPO = "lmarena-ai/arena-human-preference-55k"

TASKS = ("general", "coding", "reasoning", "planning", "vision", "speech", "system_agent")
OWN_WEIGHT = 3.0
OWN_BATTLE_WEIGHT = 5.0


def _row(prompt: str, label: str, weight: float, source: str) -> dict[str, Any]:
    return {"prompt": prompt, "label": label, "weight": weight, "source": source}


def labelled_rows(path: Path, source: str) -> tuple[list[dict], list[dict]]:
    task, docs = [], []
    for r in iter_jsonl(path):
        prompt = (r.get("prompt") or "").strip()
        if not prompt:
            continue
        if r.get("task") in TASKS:
            task.append(_row(prompt, r["task"], 1.0, source))
        if isinstance(r.get("needs_docs"), bool):
            docs.append(_row(prompt, "yes" if r["needs_docs"] else "no", 1.0, source))
    return task, docs


def own_rows(store=None) -> tuple[list[dict], list[dict]]:
    """Your corrections, from nexus.db."""
    import feedback

    task, docs = [], []
    for s in feedback.signals(store=store):
        prompt = (s.get("question") or "").strip()
        if not prompt:
            continue
        if s["kind"] == "task_override" and s.get("value") in TASKS:
            task.append(_row(prompt, s["value"], OWN_WEIGHT, "yours"))
        elif s["kind"] == "rag_override" and s.get("value") in ("always", "never"):
            docs.append(_row(prompt, "yes" if s["value"] == "always" else "no",
                             OWN_WEIGHT, "yours"))
    return task, docs


def own_strong_rows(store=None) -> list[dict]:
    """Your Arena battles between a local and a cloud model."""
    import feedback
    import providers

    rows = []
    for b in feedback.battles(store=store):
        if b["winner"] not in ("a", "b", "tie"):
            continue
        spec_a, spec_b = providers.spec_by_name(b["model_a"]), providers.spec_by_name(b["model_b"])
        if spec_a is None or spec_b is None or spec_a.is_local == spec_b.is_local:
            continue
        cloud_side = "a" if not spec_a.is_local else "b"
        label = "strong" if b["winner"] == cloud_side else "weak"
        row = _row(b["prompt"], label, OWN_BATTLE_WEIGHT, "yours")
        row["outcome"] = "tie" if b["winner"] == "tie" else label
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Public preference data (RouteLLM-style)
# ---------------------------------------------------------------------------


def _first_turn(prompt_field: Any) -> str:
    """The dataset stores prompts as a JSON list of turns."""
    if isinstance(prompt_field, list):
        turns = prompt_field
    else:
        try:
            turns = json.loads(prompt_field)
        except (TypeError, ValueError):
            return str(prompt_field or "")
    return str(turns[0]) if turns else ""


def tier_models(records: list[dict], min_battles: int = 200) -> tuple[set[str], set[str], dict]:
    """Split models into strong (top quarter by win rate) and weak (bottom
    half), using only models with enough battles to have a stable rate."""
    games: dict[str, float] = defaultdict(float)
    score: dict[str, float] = defaultdict(float)
    for r in records:
        a, b = r["model_a"], r["model_b"]
        games[a] += 1
        games[b] += 1
        if r["winner"] == "a":
            score[a] += 1
        elif r["winner"] == "b":
            score[b] += 1
        else:
            score[a] += 0.5
            score[b] += 0.5
    rates = {m: score[m] / games[m] for m in games if games[m] >= min_battles}
    ranked = sorted(rates, key=rates.get, reverse=True)
    quarter, half = max(1, len(ranked) // 4), len(ranked) // 2
    return set(ranked[:quarter]), set(ranked[half:]), rates


def strong_rows_from_records(records: list[dict], min_battles: int = 200) -> tuple[list[dict], dict]:
    strong, weak, rates = tier_models(records, min_battles)
    rows = []
    for r in records:
        a, b = r["model_a"], r["model_b"]
        if a in strong and b in weak:
            strong_side = "a"
        elif b in strong and a in weak:
            strong_side = "b"
        else:
            continue
        prompt = r["prompt"].strip()[:2000]
        if not prompt:
            continue
        # A tie means the weak model was enough. `outcome` keeps the tie for
        # evaluation: routing either way "succeeds" on a tie.
        row = _row(prompt, "strong" if r["winner"] == strong_side else "weak", 1.0, "arena55k")
        row["outcome"] = ("tie" if r["winner"] == "tie"
                          else "strong" if r["winner"] == strong_side else "weak")
        rows.append(row)
    info = {"strong_models": sorted(strong), "weak_models": sorted(weak),
            "battles_used": len(rows), "battles_total": len(records),
            "strong_win_share": round(sum(r["label"] == "strong" for r in rows) / max(len(rows), 1), 3)}
    return rows, info


def download_arena55k(limit: int | None = None) -> tuple[list[dict], dict]:
    """Fetch the public dataset (needs `pip install datasets`) and report its
    licence from the dataset card."""
    from datasets import load_dataset

    license_ = None
    try:
        from huggingface_hub import dataset_info

        card = dataset_info(ARENA55K_REPO).card_data
        license_ = getattr(card, "license", None) if card else None
    except Exception:
        pass
    ds = load_dataset(ARENA55K_REPO, split="train")
    records = []
    for i, r in enumerate(ds):
        if limit and i >= limit:
            break
        winner = ("a" if r.get("winner_model_a") else "b" if r.get("winner_model_b") else "tie")
        records.append({"prompt": _first_turn(r.get("prompt")), "model_a": r["model_a"],
                        "model_b": r["model_b"], "winner": winner})
    return records, {"license": license_, "repo": ARENA55K_REPO}


def prepare_arena55k(limit: int | None = None, path: Path | None = None) -> dict:
    path = path or ARENA55K
    records, source = download_arena55k(limit)
    rows, info = strong_rows_from_records(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    info.update(source)
    (path.with_suffix(".info.json")).write_text(json.dumps(info, indent=2))
    return info


def arena55k_rows(path: Path | None = None) -> list[dict]:
    return list(iter_jsonl(path or ARENA55K))


# ---------------------------------------------------------------------------


def task_and_docs_rows(store=None, include_own: bool = True) -> tuple[list[dict], list[dict]]:
    task, docs = labelled_rows(SEED, "seed")
    t2, d2 = labelled_rows(TEACHER, "teacher")
    task += t2
    docs += d2
    if include_own:
        try:
            t3, d3 = own_rows(store)
            task += t3
            docs += d3
        except Exception:
            pass
    return task, docs


def strong_head_rows(store=None, include_own: bool = True) -> list[dict]:
    rows = arena55k_rows()
    if include_own:
        try:
            rows += own_strong_rows(store)
        except Exception:
            pass
    return rows


def counts(rows: list[dict]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for r in rows:
        out[r["source"]] += 1
    return dict(out)
