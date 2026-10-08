"""
council.py
A model council for hard questions: several models answer, a judge compares
them, and you get one merged answer plus a map of where they agreed and
where they didn't.

Why: one model's confident answer reads the same whether it is right or
wrong. Three models that independently agree are more likely right, and a
point they disagree on is exactly where you should look twice.

  members    up to 3 different models from the request's routing chain, so
             every privacy and cost rule still applies. Cloud members run in
             parallel; local ones one after another (an 8 GB card holds one
             model at a time).
  judge      the strongest model allowed for the request (or the one set in
             [council] judge). It returns agreements, disagreements with
             each model's position, and a final merged answer.
  agreement  a model-free check alongside the judge: how similar the answers
             are to each other (embedding cosine when available, else word
             overlap), 0-100%.

If the judge fails or returns something unreadable, the answer most like
the others is used as the final one, and the UI says so.
"""

from __future__ import annotations

import json
import re
from itertools import combinations
from typing import Any

JUDGE_INSTRUCTIONS = """You are the judge of a model council. Several AI models answered the
same question independently. Compare their answers and reply with JSON only, in this shape:
{"agreements": ["a point all or most answers make"],
 "disagreements": [{"point": "what they differ on",
                    "positions": {"1": "what answer 1 says", "2": "what answer 2 says"}}],
 "final": "the best single answer for the user"}
Rules: at most 5 agreements and 5 disagreements, each one short. Only list a disagreement
when the answers really conflict or one says something the others contradict. In "final",
merge what is right in each answer and drop what is wrong; write it for the user, in plain
prose or markdown, without mentioning the council or the numbered answers."""

MAX_ANSWER_CHARS = 4000


def pick_members(chain: list[dict[str, Any]], n: int = 3) -> list[dict[str, Any]]:
    """The best `n` different models in the chain (PC actions excluded)."""
    seen, members = set(), []
    for c in chain:
        if c.get("provider") == "toolkit" or c["model"] in seen:
            continue
        seen.add(c["model"])
        members.append(c)
        if len(members) == n:
            break
    return members


def judge_messages(question: str, answers: list[str]) -> list[dict[str, str]]:
    body = [f"Question:\n{question}\n"]
    for i, text in enumerate(answers, 1):
        body.append(f"Answer {i}:\n{text[:MAX_ANSWER_CHARS]}\n")
    return [{"role": "system", "content": JUDGE_INSTRUCTIONS},
            {"role": "user", "content": "\n".join(body)}]


def parse_verdict(text: str, n_answers: int) -> dict[str, Any] | None:
    """The judge's JSON, cleaned up; None if it isn't usable."""
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    final = data.get("final")
    if not isinstance(final, str) or not final.strip():
        return None
    agreements = [str(a).strip() for a in data.get("agreements") or [] if str(a).strip()][:5]
    disagreements = []
    for d in data.get("disagreements") or []:
        if not isinstance(d, dict) or not str(d.get("point", "")).strip():
            continue
        positions = {}
        for key, value in (d.get("positions") or {}).items():
            digits = re.sub(r"\D", "", str(key))
            if digits and 1 <= int(digits) <= n_answers and str(value).strip():
                positions[int(digits)] = str(value).strip()
        disagreements.append({"point": str(d["point"]).strip(), "positions": positions})
    return {"agreements": agreements, "disagreements": disagreements[:5],
            "final": final.strip()}


# ---------------------------------------------------------------------------
# Model-free agreement
# ---------------------------------------------------------------------------


def _similarity_matrix(answers: list[str]):
    import numpy as np

    try:
        from embeddings import get_sentence_transformer

        vectors = get_sentence_transformer().encode(answers, normalize_embeddings=True,
                                                    show_progress_bar=False)
        return np.clip(vectors @ vectors.T, 0.0, 1.0), "embedding cosine"
    except Exception:
        from truth_check import content_words

        sets = [content_words(a) for a in answers]
        m = np.eye(len(answers))
        for i, j in combinations(range(len(answers)), 2):
            union = sets[i] | sets[j]
            m[i, j] = m[j, i] = len(sets[i] & sets[j]) / len(union) if union else 0.0
        return m, "word overlap"


def agreement(answers: list[str]) -> dict[str, Any]:
    """Mean pairwise similarity, and which answer is closest to the rest."""
    if len(answers) < 2:
        return {"score": None, "method": None, "central": 0}
    m, method = _similarity_matrix(answers)
    pairs = [m[i, j] for i, j in combinations(range(len(answers)), 2)]
    centrality = [(m[i].sum() - m[i, i]) / (len(answers) - 1) for i in range(len(answers))]
    return {"score": round(float(sum(pairs) / len(pairs)), 3), "method": method,
            "central": int(max(range(len(answers)), key=lambda i: centrality[i]))}
