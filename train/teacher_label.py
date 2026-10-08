"""
train/teacher_label.py
Labels prompts with a cloud model ("teacher"), to grow the router's training
data far beyond what anyone would hand-label.

    python train/teacher_label.py prompts.txt --provider gemini --model gemini-3.7-flash
    python train/teacher_label.py --from-arena55k 2000 --provider groq --model openai/gpt-oss-120b

Writes one JSON line per prompt to train/data/teacher_labels.jsonl:
{"prompt", "task", "needs_docs", "source": "teacher:<model>"}. Prompts already
labelled are skipped, so it can be stopped and resumed.

Privacy: this sends the prompts to the cloud provider. It only ever reads
the file you give it or the public dataset -- never your chats. It counts
against the provider's daily limit like any other request.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cloud  # noqa: E402
from train import data_sources  # noqa: E402

INSTRUCTIONS = """You label prompts for a router that sends each prompt to the right AI model.
Choose exactly one task:
- general: everyday questions, facts, explanations, chat
- coding: writing, fixing, explaining or reviewing code; errors and stack traces
- reasoning: comparing options, trade-offs, analysis, puzzles, arguments
- planning: plans, roadmaps, schedules, step-by-step approaches
- vision: the prompt is about an attached image, photo, chart or screenshot
- speech: the prompt is about an attached audio recording
- system_agent: organising, sorting, cleaning or finding files on the user's own computer
Also decide needs_docs: true only if answering requires the user's own private
documents or notes (e.g. "according to my notes", "what does our readme say");
false for general knowledge, even when the prompt mentions a project or a file.
Reply with JSON only, like {"task": "coding", "needs_docs": false}."""


def parse_label(text: str) -> dict | None:
    match = re.search(r"\{.*?\}", text, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    task = str(data.get("task", "")).strip().lower()
    if task not in data_sources.TASKS:
        return None
    needs = data.get("needs_docs")
    if isinstance(needs, str):
        needs = needs.strip().lower() == "true"
    return {"task": task, "needs_docs": bool(needs)}


def label_one(provider: str, model: str, prompt: str) -> dict | None:
    messages = [{"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": f"Prompt to label:\n<<<\n{prompt[:2000]}\n>>>"}]
    import cloud_client

    text = cloud_client.chat(provider, model, messages, temperature=0.0)
    cloud.record_success(provider, len(prompt), len(text))
    return parse_label(text)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("prompts", nargs="?", help="text file, one prompt per line")
    parser.add_argument("--from-arena55k", type=int, default=0, metavar="N",
                        help="label N prompts from the public Arena dataset instead")
    parser.add_argument("--provider", required=True, choices=sorted(cloud.PROVIDERS))
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default=str(data_sources.TEACHER))
    parser.add_argument("--max", type=int, default=0, help="stop after N new labels")
    args = parser.parse_args(argv)

    if args.prompts:
        prompts = [l.strip() for l in Path(args.prompts).read_text(encoding="utf-8").splitlines()
                   if l.strip()]
    elif args.from_arena55k:
        records, _ = data_sources.download_arena55k(args.from_arena55k)
        prompts = [r["prompt"].strip() for r in records if r["prompt"].strip()]
    else:
        parser.error("give a prompts file or --from-arena55k N")

    out = Path(args.out)
    done = {r.get("prompt") for r in data_sources.iter_jsonl(out)}
    status = cloud.provider_status(args.provider)
    if status["status"] != "ready":
        raise SystemExit(f"{args.provider} is not ready: {status['detail']}")

    labelled = failed = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as fh:
        for prompt in prompts:
            if prompt in done:
                continue
            if args.max and labelled >= args.max:
                break
            try:
                label = label_one(args.provider, args.model, prompt)
            except cloud.RateLimited as exc:
                print(f"stopping: {exc}")
                break
            except cloud.CloudError as exc:
                print(f"skipped one prompt: {exc}")
                failed += 1
                continue
            if label is None:
                failed += 1
                continue
            fh.write(json.dumps({"prompt": prompt, **label,
                                 "source": f"teacher:{args.model}"}, ensure_ascii=False) + "\n")
            done.add(prompt)
            labelled += 1
    print(f"labelled {labelled}, unusable {failed}, written to {out}")
    return {"labelled": labelled, "failed": failed}


if __name__ == "__main__":
    main()
