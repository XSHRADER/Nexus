"""
learned_router.py
The parts of routing NEXUS learns from data instead of hand-written rules.

Three small classifiers ("heads"), all the same kind of model:

  task      which of the 7 tasks a prompt is (general, coding, ...)
  docs      whether answering needs your own documents
  strong    whether the prompt needs a strong model (cloud) or a small local
            one will do -- trained on human preference votes, as in RouteLLM

Model: multinomial logistic regression, written in numpy so it runs on any
machine NEXUS runs on, with no extra install. Training is seconds on the
seed data and minutes on 50k public votes.

Features: hashed word unigrams and bigrams plus character 3-5-grams (robust
to typos and word forms), a few cheap signals (length, code, file paths), and
-- when sentence-transformers is installed -- the 384-dim all-MiniLM-L6-v2
embedding the router already computes. Hashing keeps the feature space a
fixed size, so a model trained on one machine loads on another.

Models are versioned in models/ (git-ignored) with their measured metrics
next to them. models/current.json points at the version in use, and
train/train_router.py only moves that pointer when the new version beats
both the hand-written rules and the previous version on the held-out set.
"""

from __future__ import annotations

import json
import math
import re
import time
import zlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from config import PROJECT_DIR

MODELS_DIR = PROJECT_DIR / "models"
CURRENT = MODELS_DIR / "current.json"

HASH_DIM = 2 ** 15
EMBED_DIM = 384
FEATURE_VERSION = 1
HEADS = ("task", "docs", "strong")


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9]+(?:['.][a-z0-9]+)*")
_CODE = re.compile(r"```|\b(def|class|import|return|function|const|var|public|void)\b|[{};]\s*$|"
                   r"\w+\(.*\)|Error\b|Exception\b", re.M)
_PATH = re.compile(r"[A-Za-z]:\\|(?:^|\s)/[\w.-]+/|\"[^\"]+\\[^\"]+\"")


def _bucket(key: str) -> tuple[int, float]:
    """Stable hash (unlike Python's hash(), which changes every run) to an
    index and a sign; the sign keeps collisions from piling up in one
    direction."""
    h = zlib.crc32(key.encode("utf-8"))
    return h % HASH_DIM, (1.0 if (h >> 31) & 1 else -1.0)


def hashed_features(text: str) -> dict[int, float]:
    lower = text.lower()
    words = _WORD.findall(lower)
    counts: dict[int, float] = {}

    def add(key: str, weight: float = 1.0) -> None:
        idx, sign = _bucket(key)
        counts[idx] = counts.get(idx, 0.0) + sign * weight

    for w in words:
        add("w:" + w)
        padded = f" {w} "
        for n in (3, 4, 5):
            for i in range(len(padded) - n + 1):
                add("c:" + padded[i:i + n], 0.3)
    for a, b in zip(words, words[1:]):
        add(f"b:{a}_{b}", 0.7)
    if words:
        add("first:" + words[0], 1.5)       # "write ...", "sort ...", "why ..."
        add("first2:" + "_".join(words[:2]), 1.0)
    # Cheap shape signals.
    n = len(words)
    add("len:" + ("1-3" if n <= 3 else "4-10" if n <= 10 else "11-30" if n <= 30 else "30+"))
    if _CODE.search(text):
        add("has:code", 2.0)
    if _PATH.search(text):
        add("has:path", 2.0)
    if text.rstrip().endswith("?"):
        add("has:question")
    # Sublinear term frequency, then unit length.
    feats = {i: math.copysign(math.log1p(abs(v)), v) for i, v in counts.items() if v}
    norm = math.sqrt(sum(v * v for v in feats.values())) or 1.0
    return {i: v / norm for i, v in feats.items()}


@lru_cache(maxsize=1)
def _embedder():
    from embeddings import get_sentence_transformer

    return get_sentence_transformer()


def embeddings_available() -> bool:
    try:
        _embedder()
        return True
    except Exception:
        return False


@dataclass
class Matrix:
    """Rows of sparse features in CSR form: just what the trainer needs."""

    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    n_features: int

    @property
    def n_rows(self) -> int:
        return len(self.indptr) - 1

    def rows(self, order: np.ndarray) -> "Matrix":
        indptr, indices, data = [0], [], []
        for r in order:
            a, b = self.indptr[r], self.indptr[r + 1]
            indices.append(self.indices[a:b])
            data.append(self.data[a:b])
            indptr.append(indptr[-1] + (b - a))
        return Matrix(np.asarray(indptr, dtype=np.int64),
                      np.concatenate(indices) if indices else np.zeros(0, np.int64),
                      np.concatenate(data) if data else np.zeros(0, np.float32),
                      self.n_features)


def featurize(texts: Sequence[str], use_embeddings: bool = False,
              embed_weight: float = 1.0) -> Matrix:
    n_features = HASH_DIM + (EMBED_DIM if use_embeddings else 0)
    vectors = None
    if use_embeddings:
        vectors = _embedder().encode(list(texts), normalize_embeddings=True, batch_size=64,
                                     show_progress_bar=False)
    indptr, indices, data = [0], [], []
    for r, text in enumerate(texts):
        feats = hashed_features(text)
        idx = np.fromiter(feats.keys(), dtype=np.int64, count=len(feats))
        val = np.fromiter(feats.values(), dtype=np.float32, count=len(feats))
        if vectors is not None:
            idx = np.concatenate([idx, HASH_DIM + np.arange(EMBED_DIM, dtype=np.int64)])
            val = np.concatenate([val, np.asarray(vectors[r], dtype=np.float32) * embed_weight])
        indices.append(idx)
        data.append(val)
        indptr.append(indptr[-1] + len(idx))
    return Matrix(np.asarray(indptr, dtype=np.int64),
                  np.concatenate(indices) if indices else np.zeros(0, np.int64),
                  np.concatenate(data) if data else np.zeros(0, np.float32), n_features)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class LinearModel:
    """Multinomial logistic regression trained with Adam on sparse rows."""

    def __init__(self, classes: Sequence[str], n_features: int, use_embeddings: bool = False):
        self.classes = list(classes)
        self.n_features = n_features
        self.use_embeddings = use_embeddings
        self.W = np.zeros((n_features, len(self.classes)), dtype=np.float32)
        self.b = np.zeros(len(self.classes), dtype=np.float32)

    # -- inference -------------------------------------------------------------

    def _logits(self, X: Matrix, start: int = 0, stop: int | None = None) -> np.ndarray:
        stop = X.n_rows if stop is None else stop
        a, b = X.indptr[start], X.indptr[stop]
        contrib = X.data[a:b, None] * self.W[X.indices[a:b]]
        starts = X.indptr[start:stop] - a
        out = np.zeros((stop - start, len(self.classes)), dtype=np.float32)
        nonempty = X.indptr[start + 1:stop + 1] > X.indptr[start:stop]
        if contrib.shape[0]:
            sums = np.add.reduceat(contrib, starts[nonempty], axis=0)
            out[nonempty] = sums
        return out + self.b

    def predict_proba(self, X: Matrix) -> np.ndarray:
        return _softmax(self._logits(X))

    def predict(self, X: Matrix) -> list[str]:
        return [self.classes[i] for i in self.predict_proba(X).argmax(axis=1)]

    # -- training --------------------------------------------------------------

    def fit(self, X: Matrix, y: Sequence[str], weights: Sequence[float] | None = None,
            epochs: int = 40, batch_size: int = 64, lr: float = 0.05, l2: float = 1e-4,
            balance: bool = True, seed: int = 0) -> "LinearModel":
        rng = np.random.default_rng(seed)
        index = {c: i for i, c in enumerate(self.classes)}
        Y = np.array([index[v] for v in y], dtype=np.int64)
        w = np.ones(len(Y), dtype=np.float32) if weights is None else np.asarray(weights, np.float32)
        if balance:  # every class counts equally, however rare
            counts = np.bincount(Y, minlength=len(self.classes)).astype(np.float32)
            w = w * (len(Y) / (len(self.classes) * np.maximum(counts, 1)))[Y]
        mW, vW = np.zeros_like(self.W), np.zeros_like(self.W)
        mb, vb = np.zeros_like(self.b), np.zeros_like(self.b)
        beta1, beta2, eps, t = 0.9, 0.999, 1e-8, 0
        for _ in range(epochs):
            order = rng.permutation(X.n_rows)
            Xs, Ys, ws = X.rows(order), Y[order], w[order]
            for start in range(0, Xs.n_rows, batch_size):
                stop = min(start + batch_size, Xs.n_rows)
                P = _softmax(self._logits(Xs, start, stop))
                P[np.arange(stop - start), Ys[start:stop]] -= 1.0
                G = P * ws[start:stop, None] / max(ws[start:stop].sum(), 1e-6)
                a, b = Xs.indptr[start], Xs.indptr[stop]
                row_of = np.repeat(np.arange(stop - start), np.diff(Xs.indptr[start:stop + 1]))
                gW = np.zeros_like(self.W)
                np.add.at(gW, Xs.indices[a:b], Xs.data[a:b, None] * G[row_of])
                gW += l2 * self.W
                gb = G.sum(axis=0)
                t += 1
                mW = beta1 * mW + (1 - beta1) * gW
                vW = beta2 * vW + (1 - beta2) * gW * gW
                mb = beta1 * mb + (1 - beta1) * gb
                vb = beta2 * vb + (1 - beta2) * gb * gb
                corr = math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                self.W -= (lr * corr * mW / (np.sqrt(vW) + eps)).astype(np.float32)
                self.b -= (lr * corr * mb / (np.sqrt(vb) + eps)).astype(np.float32)
        return self

    # -- persistence -----------------------------------------------------------

    def to_arrays(self, prefix: str) -> dict[str, np.ndarray]:
        # Only rows that were ever touched are stored, so files stay small.
        used = np.flatnonzero(np.abs(self.W).sum(axis=1) > 0)
        return {f"{prefix}_rows": used.astype(np.int64), f"{prefix}_W": self.W[used],
                f"{prefix}_b": self.b}

    @classmethod
    def from_arrays(cls, arrays, prefix: str, classes: Sequence[str], n_features: int,
                    use_embeddings: bool) -> "LinearModel":
        model = cls(classes, n_features, use_embeddings)
        model.W[arrays[f"{prefix}_rows"]] = arrays[f"{prefix}_W"]
        model.b = arrays[f"{prefix}_b"].astype(np.float32)
        return model


# ---------------------------------------------------------------------------
# A trained router: up to three heads that share one feature space
# ---------------------------------------------------------------------------


class LearnedRouter:
    def __init__(self, heads: dict[str, LinearModel], meta: dict[str, Any]):
        self.heads = heads
        self.meta = meta
        self.use_embeddings = bool(meta.get("use_embeddings"))

    @property
    def version(self) -> str:
        return self.meta.get("version", "?")

    def _features(self, text: str) -> Matrix:
        return featurize([text], use_embeddings=self.use_embeddings)

    def predict(self, text: str) -> dict[str, dict[str, float]]:
        """{head: {class: probability}} for every head this router has."""
        X = self._features(text)
        return {
            name: dict(zip(head.classes, map(float, head.predict_proba(X)[0])))
            for name, head in self.heads.items()
        }

    def save(self, path: Path) -> None:
        arrays: dict[str, np.ndarray] = {}
        for name, head in self.heads.items():
            arrays.update(head.to_arrays(name))
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path, meta: dict[str, Any]) -> "LearnedRouter":
        arrays = np.load(path)
        n_features = HASH_DIM + (EMBED_DIM if meta.get("use_embeddings") else 0)
        heads = {
            name: LinearModel.from_arrays(arrays, name, info["classes"], n_features,
                                          bool(meta.get("use_embeddings")))
            for name, info in meta.get("heads", {}).items()
        }
        return cls(heads, meta)


def next_version() -> str:
    MODELS_DIR.mkdir(exist_ok=True)
    existing = [int(m.group(1)) for p in MODELS_DIR.glob("router_v*.npz")
                if (m := re.match(r"router_v(\d+)\.npz", p.name))]
    return f"v{max(existing, default=0) + 1}"


def save_version(router: LearnedRouter, make_current: bool) -> Path:
    MODELS_DIR.mkdir(exist_ok=True)
    version = router.meta["version"]
    router.save(MODELS_DIR / f"router_{version}.npz")
    (MODELS_DIR / f"router_{version}.json").write_text(json.dumps(router.meta, indent=2))
    if make_current:
        CURRENT.write_text(json.dumps({"version": version, "since": time.time()}))
    return MODELS_DIR / f"router_{version}.json"


def current_meta() -> dict[str, Any] | None:
    if not CURRENT.exists():
        return None
    try:
        version = json.loads(CURRENT.read_text())["version"]
        return json.loads((MODELS_DIR / f"router_{version}.json").read_text())
    except (OSError, ValueError, KeyError):
        return None


def load_current() -> LearnedRouter | None:
    """The router version in use, or None (then the rules route alone)."""
    meta = current_meta()
    if meta is None:
        return None
    if meta.get("feature_version") != FEATURE_VERSION:
        return None
    if meta.get("use_embeddings") and not embeddings_available():
        return None
    try:
        return LearnedRouter.load(MODELS_DIR / f"router_{meta['version']}.npz", meta)
    except (OSError, KeyError, ValueError):
        return None


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
