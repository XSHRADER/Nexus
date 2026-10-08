"""
demos/mock_judge.py
A stand-in for a council judge, shared by mock_ollama.py and mock_cloud.py.

It reads the numbered answers it was given and returns the JSON verdict
council.py expects, built from simple text comparisons. Not a real judge:
it exists so the council's plumbing and UI can be demonstrated and tested.
"""

import json
import re

MARKER = "You are the judge of a model council"
STOP = set("a an the and or of to in on is are it its this that with for as by be you your "
           "it's what how into each when at from than".split())


def _sentences(text: str) -> list[str]:
    text = re.sub(r"_\(mock[^)]*\)_", "", text).strip()
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _words(text: str) -> set[str]:
    text = re.sub(r"_\(mock[^)]*\)_", "", text)
    return {w for w in re.findall(r"[a-z][a-z'-]+", text.lower()) if w not in STOP and len(w) > 3}


def verdict(user_content: str) -> str:
    answers = {}
    if "Answer 1:" in user_content:
        chunks = re.split(r"Answer (\d+):\n", user_content)
        for i in range(1, len(chunks) - 1, 2):
            answers[int(chunks[i])] = chunks[i + 1].strip()
    if not answers:
        return json.dumps({"agreements": [], "disagreements": [], "final": "No answers given."})
    common = set.intersection(*(_words(a) for a in answers.values()))
    agreements = ([f"All {len(answers)} answers talk about: {', '.join(sorted(common)[:4])}."]
                  if common else [])
    disagreements = []
    lengths = {k: len(_sentences(v)) for k, v in answers.items()}
    if max(lengths.values()) - min(lengths.values()) >= 2:
        disagreements.append({"point": "How much detail the question needs",
                              "positions": {str(k): (_sentences(v) or [v])[0][:140]
                                            for k, v in answers.items()}})
    speed = {k: bool(re.search(r"constant time|o\(1\)", v, re.I)) for k, v in answers.items()}
    if any(speed.values()) and not all(speed.values()):
        disagreements.append({"point": "How fast lookups are",
                              "positions": {str(k): ("lookups take constant time on average"
                                                     if s else "doesn't say") for k, s in speed.items()}})
    best = max(answers.values(), key=lambda a: len(_sentences(a)))
    final = " ".join(_sentences(best)) + "\n\n_(mock judge · merged from " \
            f"{len(answers)} answers)_"
    return json.dumps({"agreements": agreements, "disagreements": disagreements, "final": final})
