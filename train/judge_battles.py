"""
train/judge_battles.py
Find out, prompt by prompt, whether a strong (cloud) model actually answers
better than a small local one -- without waiting for weeks of Arena votes.

For each prompt a local model and a cloud model both answer, and a panel of
judge models grades the two answers blind (they see "Answer A" and "Answer B"
in a random order, never the model names). The judges are not asked which
answer they prefer: a judge asked that picks the more polished one nearly
every time, which says nothing about whether the cloud was needed. They are
asked whether each answer is acceptable on its own. The cloud only "won" a
prompt when the local answer was not acceptable and the cloud answer was.

Every judge on the panel must reach the same outcome. Where they disagree the
prompt is recorded as "disputed" and is not used for training. One row per
prompt:

    {"prompt", "label": "strong" | "weak" | null,
     "outcome": "strong" | "weak" | "tie" | "disputed",
     "weak_ok", "strong_ok", "weak_model", "strong_model", "judges", "reason",
     "source": "judged"}

`label` is what the router's "needs a strong model" head trains on: "strong"
only when the cloud answer was acceptable and the local one was not. A tie
(both acceptable, or neither) means sending it to the cloud gained nothing.

These are models' judgements, not yours, and by default one of the two judges
is the cloud model itself, which is why a second, different model has to
agree. Your own Arena votes between a local and a cloud model are a better
signal and count five times as much (train/data_sources.py); this file is
what the head learns from until you have enough of them.

It sends every prompt to a cloud provider, so use it on prompts you are happy
to share. It is resumable: prompts already in the output are skipped.

    python train/judge_battles.py prompts.txt
    python train/judge_battles.py prompts.txt --weak llama3.1:8b \\
        --strong groq:openai/gpt-oss-120b \\
        --judge groq:qwen/qwen3.8-27b --judge groq:openai/gpt-oss-120b
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexus import cloud, cloud_client, ollama  # noqa: E402
from train import data_sources  # noqa: E402

DEFAULT_JUDGES = ["groq:qwen/qwen3.8-27b", "groq:openai/gpt-oss-120b"]
ANSWER_INSTRUCTIONS = "Answer the question directly and concisely, in at most 150 words."
JUDGE_INSTRUCTIONS = (
    'You are grading two answers to the same question, each on its own. An answer is "ok" if '
    "it is factually correct, actually answers what was asked, and has no serious error or "
    "omission. Differences in style, length, depth, formatting or extra detail never make an "
    'answer "not_ok": only a wrong result, a real mistake, or failing to answer does. Work '
    "out the right answer yourself before grading. Reply with JSON only: "
    '{"A": "ok" | "not_ok", "B": "ok" | "not_ok", "reason": "<at most 15 words: what is '
    'wrong, or both fine>"}.'
)
MAX_ANSWER_CHARS = 2500
RETRIES = 5


def parse_verdict(text: str) -> dict | None:
    """{"a": bool, "b": bool, "reason": str} from a judge's reply (True = acceptable)."""
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    grades = {}
    for side in ("A", "B"):
        grade = str(data.get(side, "")).strip().lower().replace(" ", "_")
        if grade not in ("ok", "not_ok"):
            return None
        grades[side.lower()] = grade == "ok"
    return {**grades, "reason": str(data.get("reason", "")).strip()[:140]}


def strong_first(prompt: str) -> bool:
    """Which answer is shown as "A": stable per prompt, and half and half overall."""
    return zlib.crc32(prompt.encode("utf-8")) % 2 == 0


def outcome_of(strong_ok: bool, weak_ok: bool) -> str:
    """Who was needed: "strong" or "weak" when only that one was acceptable,
    "tie" when both were (or neither was)."""
    if strong_ok == weak_ok:
        return "tie"
    return "strong" if strong_ok else "weak"


def _cloud(spec: str, messages: list[dict], temperature: float) -> str:
    """One cloud call, waiting out rate limits and brief outages."""
    provider, _, model = spec.partition(":")
    for attempt in range(RETRIES):
        try:
            return cloud_client.chat(provider, model, messages, temperature=temperature)
        except (cloud.RateLimited, cloud.Offline) as exc:
            if attempt == RETRIES - 1:
                raise
            time.sleep(min(60.0, getattr(exc, "retry_after", None) or 10.0 * (attempt + 1)))
    raise RuntimeError("unreachable")


def combine(verdicts: list[dict], strong_is_a: bool) -> dict:
    """The panel's result: an outcome only when every judge reached the same one."""
    graded = [((v["a"], v["b"]) if strong_is_a else (v["b"], v["a"])) for v in verdicts]
    outcomes = {outcome_of(strong_ok, weak_ok) for strong_ok, weak_ok in graded}
    reason = " / ".join(v["reason"] for v in verdicts if v["reason"])[:280]
    if len(outcomes) != 1:
        return {"label": None, "outcome": "disputed", "weak_ok": None, "strong_ok": None,
                "reason": reason}
    outcome = outcomes.pop()
    return {"label": "strong" if outcome == "strong" else "weak", "outcome": outcome,
            "strong_ok": all(s for s, _ in graded), "weak_ok": all(w for _, w in graded),
            "reason": reason}


def battle(prompt: str, weak: str, strong: str, judges: list[str]) -> dict | None:
    ask = [{"role": "system", "content": ANSWER_INSTRUCTIONS}, {"role": "user", "content": prompt}]
    weak_answer = ollama.chat(weak, ask, temperature=0.2).text[:MAX_ANSWER_CHARS]
    strong_answer = _cloud(strong, ask, 0.2)[:MAX_ANSWER_CHARS]
    strong_is_a = strong_first(prompt)
    a, b = (strong_answer, weak_answer) if strong_is_a else (weak_answer, strong_answer)
    question = [{"role": "system", "content": JUDGE_INSTRUCTIONS},
                {"role": "user", "content": f"Question:\n{prompt}\n\nAnswer A:\n{a}\n\nAnswer B:\n{b}"}]
    verdicts = [parse_verdict(_cloud(judge, question, 0.0)) for judge in judges]
    if not all(verdicts):
        return None  # a judge's reply could not be read
    return {"prompt": prompt, **combine(verdicts, strong_is_a), "weak_model": weak,
            "strong_model": strong, "judges": judges, "weight": 1.0, "source": "judged"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("prompts", help="text file, one prompt per line")
    parser.add_argument("--weak", default="llama3.1:8b", help="local Ollama model")
    parser.add_argument("--strong", default="groq:openai/gpt-oss-120b", help="provider:model")
    parser.add_argument("--judge", action="append",
                        help="provider:model; repeat for a panel, all of whom must agree "
                             f"(default: {' and '.join(DEFAULT_JUDGES)})")
    parser.add_argument("--out", default=str(data_sources.JUDGED))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    judges = args.judge or DEFAULT_JUDGES

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {data_sources.normalise(r["prompt"]) for r in data_sources.iter_jsonl(out)} if out.exists() else set()
    held = data_sources.held_out_prompts()
    prompts = [line.strip() for line in Path(args.prompts).read_text(encoding="utf-8").splitlines()
               if line.strip()]
    todo = [p for p in dict.fromkeys(prompts)
            if data_sources.normalise(p) not in done and data_sources.normalise(p) not in held]
    todo = todo[: args.limit] if args.limit else todo
    print(f"{len(todo)} prompt(s) to judge ({len(done)} already done); "
          f"{args.weak} vs {args.strong}, judged by {' + '.join(judges)}", flush=True)

    counts = {"strong": 0, "weak": 0, "tie": 0, "disputed": 0, "failed": 0}
    with open(out, "a", encoding="utf-8", newline="\n") as fh:
        for i, prompt in enumerate(todo, 1):
            try:
                row = battle(prompt, args.weak, args.strong, judges)
            except Exception as exc:
                row = None
                print(f"  [{i}/{len(todo)}] failed: {str(exc)[:100]}", flush=True)
            if row is None:
                counts["failed"] += 1
            else:
                counts[row["outcome"]] += 1
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
            if i % 10 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] cloud needed {counts['strong']}, only local right "
                      f"{counts['weak']}, no difference {counts['tie']}, judges disagreed "
                      f"{counts['disputed']}, failed {counts['failed']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
