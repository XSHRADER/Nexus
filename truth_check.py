"""
truth_check.py
Checks an answer, sentence by sentence, against your own documents.

Each factual sentence gets one of three labels:

    supported     a passage in your files says the same thing      (green)
    not_found     nothing in your files says it either way          (yellow)
    contradicted  a passage in your files says something different  (red)

and the answer gets a trust score: the share of checked sentences that are
supported. "Not found" is not "false" -- it means the claim rests on the
model's own knowledge, which is exactly what you'd want to know.

How a sentence is judged
  1. Evidence: the chunks retrieved for the answer, when there were any.
     Otherwise (or in addition, for an answer that didn't use documents) the
     best few passages from your documents for that sentence.
  2. Each (passage, sentence) pair is scored by a natural-language-inference
     (NLI) model -- a small cross-encoder that outputs P(entailment),
     P(contradiction), P(neutral). It runs on CPU like the reranker.
  3. The sentence takes its best support and its strongest contradiction
     across passages; thresholds in nexus.toml decide the label.

Without sentence-transformers or the NLI model, a keyword checker stands in:
word overlap for support, and mismatched numbers or a flipped negation for
contradiction. It is labelled "approximate" wherever it's shown, and
eval_truth.py measures both on the same labelled set.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from config import PROJECT_DIR, get_settings

DOCS_DIR = PROJECT_DIR / "documents"

SUPPORTED, NOT_FOUND, CONTRADICTED = "supported", "not_found", "contradicted"
LABELS = (SUPPORTED, NOT_FOUND, CONTRADICTED)

MIN_CLAIM_WORDS = 4

# ---------------------------------------------------------------------------
# Splitting an answer into checkable sentences
# ---------------------------------------------------------------------------

# A sentence ends at . ! ? when what follows starts a new sentence. "3.5",
# "e.g. the" and "localhost:8080." mid-line don't count as endings.
_SENTENCE_END = re.compile(r"[.!?]+(?=\s+[\"“(\[]?[A-Z0-9]|\s*$)")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_SKIP_START = re.compile(
    r"^(hi|hello|hey|sure|of course|certainly|great question|here(?:'s| is| are)|"
    r"let me know|i hope|feel free|in summary|to summari[sz]e|thanks|thank you)\b",
    re.I,
)
_MARKDOWN = re.compile(r"(\*\*|__|`|\*|_)")


@dataclass
class Claim:
    text: str           # cleaned sentence sent to the scorer
    start: int          # offsets into the original answer, for highlighting
    end: int
    label: str = NOT_FOUND
    score: float = 0.0  # probability behind the label
    source: str | None = None
    evidence: str | None = None


def _clean(sentence: str) -> str:
    return re.sub(r"\s+", " ", _MARKDOWN.sub("", sentence)).strip()


def _checkable(sentence: str) -> bool:
    text = _clean(sentence)
    if len(text.split()) < MIN_CLAIM_WORDS:
        return False
    if text.endswith("?") or text.endswith(":"):
        return False
    if text.startswith(("(", "_(")) and text.endswith(")"):  # asides, footers
        return False
    return not _SKIP_START.match(text)


def split_claims(answer: str) -> list[Claim]:
    """Factual sentences in `answer`, with their character offsets.

    Code blocks, headings, table rows, questions, greetings and very short
    fragments are skipped: they aren't claims a document could back up.
    """
    claims: list[Claim] = []
    in_code = False
    offset = 0
    for line in answer.splitlines(keepends=True):
        line_start = offset
        offset += len(line)
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not stripped or stripped.startswith(("#", "|", ">")):
            continue
        bullet = _BULLET.match(line)
        body_start = bullet.end() if bullet else len(line) - len(line.lstrip())
        body = line[body_start:].rstrip("\n")
        cursor = 0
        ends = [m.end() for m in _SENTENCE_END.finditer(body)]
        if not ends or ends[-1] < len(body.rstrip()):
            ends.append(len(body.rstrip()))
        for end in ends:
            piece = body[cursor:end]
            lead = len(piece) - len(piece.lstrip())
            if _checkable(piece):
                start = line_start + body_start + cursor + lead
                claims.append(Claim(text=_clean(piece), start=start,
                                    end=line_start + body_start + end))
            cursor = end
    return claims


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

_STOP = set("""a an the and or but if of to in on at by for with from as is are was were be been
being it its this that these those there their they them he she we you your our i my me
do does did done can could will would should may might must have has had not no yes so
than then which who whom what when where why how all any each some such into about over
also very more most only just using use used uses""".split())


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z][a-z0-9.+-]*[a-z0-9]|[a-z]", text.lower())
    return {_stem(w.strip(".")) for w in words if w not in _STOP and len(w) > 1}


def numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def overlap(claim: str, passage: str) -> float:
    """Share of the claim's content words that the passage contains."""
    want = content_words(claim)
    if not want:
        return 0.0
    return len(want & content_words(passage)) / len(want)


@dataclass
class Passage:
    source: str
    text: str


def _split_passages(source: str, text: str, max_chars: int = 700) -> list[Passage]:
    out: list[Passage] = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        while len(block) > max_chars:
            cut = block.rfind(". ", 0, max_chars)
            cut = cut + 1 if cut > max_chars // 3 else max_chars
            out.append(Passage(source, block[:cut].strip()))
            block = block[cut:].strip()
        if block:
            out.append(Passage(source, block))
    return out


def _docs_signature(folder: Path) -> tuple:
    if not folder.exists():
        return ()
    return tuple(sorted((str(p), p.stat().st_mtime) for p in folder.rglob("*")
                        if p.is_file() and p.suffix.lower() in (".txt", ".md", ".pdf", ".docx")))


@lru_cache(maxsize=2)
def _document_passages(signature: tuple) -> tuple[Passage, ...]:
    from loaders import load_document

    passages: list[Passage] = []
    for path, _mtime in signature:
        try:
            text = load_document(path)
        except Exception:
            continue
        passages.extend(_split_passages(Path(path).name, text))
    return tuple(passages)


def document_passages(folder: Path = DOCS_DIR) -> tuple[Passage, ...]:
    """Every passage in the documents folder; cached until a file changes."""
    return _document_passages(_docs_signature(folder))


def focus_excerpt(passage: str, claim: str, max_chars: int = 280) -> str:
    """The part of a passage that bears on the claim, for showing as evidence.

    The scorer reads the whole passage; a person shouldn't have to. Picks the
    line or sentence sharing the most words with the claim -- counting
    numbers, so a contradicting figure is what gets shown -- and adds its
    neighbours while they fit.
    """
    units = [u.strip() for u in re.split(r"(?<=[.!?])\s+|\n+", passage) if u.strip()]
    if not units or len(passage) <= max_chars:
        return passage.strip()
    want = content_words(claim)

    def relevance(unit: str) -> float:
        words = content_words(unit)
        return len(want & words) + 0.5 * bool(numbers(unit) and numbers(claim))

    best = max(range(len(units)), key=lambda i: relevance(units[i]))
    lo = hi = best
    text = units[best]
    while True:
        grew = False
        for j in (hi + 1, lo - 1):
            if 0 <= j < len(units) and len(text) + len(units[j]) + 1 <= max_chars:
                if j > hi:
                    text, hi = f"{text} {units[j]}", j
                else:
                    text, lo = f"{units[j]} {text}", j
                grew = True
        if not grew:
            break
    return ("… " if lo > 0 else "") + text[:max_chars] + (" …" if hi < len(units) - 1 else "")


def best_passages(claim: str, passages, k: int) -> list[Passage]:
    scored = [(overlap(claim, p.text), p) for p in passages]
    scored = [(s, p) for s, p in scored if s > 0]
    scored.sort(key=lambda sp: sp[0], reverse=True)
    return [p for _, p in scored[:k]]


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------


class Scorer(Protocol):
    name: str
    approximate: bool

    def score(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        """(premise, hypothesis) -> {"entailment", "contradiction", "neutral"}."""


_NEGATION = re.compile(r"\b(not|no|never|none|cannot|can't|doesn't|don't|isn't|aren't|"
                       r"won't|without)\b", re.I)


class KeywordScorer:
    """Baseline without any model: word overlap for support; a mismatched
    number or a flipped negation in an otherwise matching passage for
    contradiction."""

    name = "keyword"
    approximate = True

    def score(self, pairs):
        out = []
        for premise, claim in pairs:
            words = content_words(re.sub(r"\d+(?:\.\d+)?", " ", claim))
            premise_words = content_words(premise)
            ov = len(words & premise_words) / len(words) if words else 0.0
            contradiction = 0.0
            claim_nums, premise_nums = numbers(claim), numbers(premise)
            if ov >= 0.5 and claim_nums and premise_nums and not claim_nums <= premise_nums:
                contradiction = 0.6 + 0.4 * ov
            elif ov >= 0.7 and bool(_NEGATION.search(claim)) != bool(_NEGATION.search(premise)):
                contradiction = 0.55 + 0.3 * ov
            entailment = 0.0 if contradiction else (ov if ov >= 0.6 else ov * 0.5)
            neutral = max(0.0, 1.0 - entailment - contradiction)
            out.append({"entailment": entailment, "contradiction": contradiction,
                        "neutral": neutral})
        return out


class NLIScorer:
    """A cross-encoder trained on natural-language inference."""

    approximate = False

    def __init__(self, model_name: str):
        from embeddings import get_cross_encoder

        self.name = model_name
        self.model = get_cross_encoder(model_name)
        id2label = {}
        try:
            id2label = dict(self.model.model.config.id2label)
        except AttributeError:
            pass
        # The sentence-transformers NLI cross-encoders use this order.
        default = {0: "contradiction", 1: "entailment", 2: "neutral"}
        self.labels = [str((id2label or default).get(i, default[i])).lower() for i in range(3)]

    def score(self, pairs):
        if not pairs:
            return []
        logits = self.model.predict(list(pairs), batch_size=16, show_progress_bar=False)
        out = []
        for row in logits:
            row = [float(x) for x in row]
            top = max(row)
            exps = [math.exp(x - top) for x in row]
            total = sum(exps)
            probs = {label: e / total for label, e in zip(self.labels, exps)}
            out.append({"entailment": probs.get("entailment", 0.0),
                        "contradiction": probs.get("contradiction", 0.0),
                        "neutral": probs.get("neutral", 0.0)})
        return out


_scorer_cache: dict[str, Any] = {}


def get_scorer(method: str | None = None):
    """The configured scorer, falling back to keywords if the model can't load."""
    settings = get_settings()
    method = (method or settings.truth_method).lower()
    key = f"{method}:{settings.truth_model}"
    if key in _scorer_cache:
        return _scorer_cache[key]
    scorer: Any = KeywordScorer()
    if method in ("auto", "nli"):
        try:
            scorer = NLIScorer(settings.truth_model)
        except Exception:
            if method == "nli":
                raise
    _scorer_cache[key] = scorer
    return scorer


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------


def label_for(probs: dict[str, float], support_threshold: float,
              contradict_threshold: float) -> tuple[str, float]:
    if probs["entailment"] >= support_threshold and probs["entailment"] >= probs["contradiction"]:
        return SUPPORTED, probs["entailment"]
    if probs["contradiction"] >= contradict_threshold:
        return CONTRADICTED, probs["contradiction"]
    return NOT_FOUND, probs.get("neutral", 0.0)


@dataclass
class TruthReport:
    claims: list[Claim] = field(default_factory=list)
    method: str = ""
    approximate: bool = False
    evidence: str = ""          # "answer sources" | "your documents" | "none"
    elapsed: float = 0.0
    status: str = "ok"          # ok | nothing_to_check | no_documents | off

    @property
    def counts(self) -> dict[str, int]:
        return {label: sum(c.label == label for c in self.claims) for label in LABELS}

    @property
    def trust(self) -> float | None:
        return (self.counts[SUPPORTED] / len(self.claims)) if self.claims else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "method": self.method,
            "approximate": self.approximate,
            "evidence": self.evidence,
            "trust": None if self.trust is None else round(self.trust, 3),
            "counts": self.counts,
            "elapsed": round(self.elapsed, 3),
            "claims": [
                {"text": c.text, "start": c.start, "end": c.end, "label": c.label,
                 "score": round(c.score, 3), "source": c.source,
                 "evidence": focus_excerpt(c.evidence, c.text) if c.evidence else None}
                for c in self.claims
            ],
        }


def check(
    answer: str,
    sources: list[dict[str, Any]] | None = None,
    scorer=None,
    passages=None,
) -> dict[str, Any]:
    """Check `answer` against `sources` (the chunks retrieved for it) plus
    the best matching passages from your documents."""
    started = time.monotonic()
    settings = get_settings()
    if settings.truth_method == "off" and scorer is None:
        return TruthReport(status="off").to_dict()

    claims = split_claims(answer or "")[: settings.truth_max_claims]
    report = TruthReport(claims=claims)
    if not claims:
        report.status = "nothing_to_check"
        return report.to_dict()

    given = [Passage(s.get("source", "unknown"), s.get("text", ""))
             for s in (sources or []) if s.get("text")]
    if passages is None:
        try:
            passages = document_passages()
        except Exception:
            passages = ()
    pool = list(given) + [p for p in passages
                          if not any(g.text == p.text for g in given)]
    report.evidence = "answer sources" if given else ("your documents" if pool else "none")
    if not pool:
        report.status = "no_documents"
        report.elapsed = time.monotonic() - started
        for c in claims:
            c.label = NOT_FOUND
        return report.to_dict()

    scorer = scorer or get_scorer()
    report.method = scorer.name
    report.approximate = bool(scorer.approximate)

    pairs: list[tuple[str, str]] = []
    owners: list[tuple[int, Passage]] = []
    for i, claim in enumerate(claims):
        for passage in best_passages(claim.text, pool, settings.evidence_per_claim):
            pairs.append((passage.text, claim.text))
            owners.append((i, passage))
    results = scorer.score(pairs) if pairs else []

    best: dict[int, tuple[str, float, Passage | None]] = {}
    for (i, passage), probs in zip(owners, results):
        label, prob = label_for(probs, settings.support_threshold,
                                settings.contradict_threshold)
        rank = {SUPPORTED: 2, CONTRADICTED: 1, NOT_FOUND: 0}[label]
        current = best.get(i)
        current_rank = {SUPPORTED: 2, CONTRADICTED: 1, NOT_FOUND: 0}[current[0]] if current else -1
        # Support anywhere wins (the claim *is* in your files); otherwise the
        # strongest contradiction; otherwise not found.
        if rank > current_rank or (rank == current_rank and prob > current[1]):
            best[i] = (label, prob, passage if label != NOT_FOUND else None)

    for i, claim in enumerate(claims):
        label, prob, passage = best.get(i, (NOT_FOUND, 0.0, None))
        claim.label, claim.score = label, prob
        if passage is not None:
            claim.source, claim.evidence = passage.source, passage.text

    report.elapsed = time.monotonic() - started
    return report.to_dict()
