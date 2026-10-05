"""
retrieve.py
Hybrid retrieval: vector similarity (Chroma) + keyword search (BM25), fused
with Reciprocal Rank Fusion, then reranked by a cross-encoder.

Why RRF rather than a weighted sum of the two scores: cosine similarity and
BM25 live on different, query-dependent scales, so `0.65 * cosine + 0.35 *
bm25` compares numbers that don't mean the same thing between one query and
the next. RRF throws the magnitudes away and fuses on *rank*, which is what
those two arms actually agree on.

Three stages, narrowing each time:
    vector top-40 + bm25 top-40  ->  RRF  ->  cross-encoder top-25  ->  top_k

Usage:
    python -m nexus.retrieve        # interactive retrieval-only test
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi

from nexus import config
from nexus.embeddings import get_cross_encoder, get_sentence_transformer

log = logging.getLogger(__name__)

# Written by ingest.py at the end of every run that changes the index; its
# mtime is how a long-lived retriever notices that it should reload.
STAMP_NAME = "file_hashes.json"

_WORD = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word/identifier tokens.

    Splitting on non-word characters rather than whitespace means a query for
    `router` still matches `router.py`, which is most of the point of keeping
    a keyword arm next to the vector one.
    """
    return _WORD.findall(text.lower())


@dataclass
class _Snapshot:
    """Everything one query reads, swapped in as a whole on refresh, so a
    query running in another Streamlit session never sees half a rebuild."""

    collection: object | None = None
    ids: list[str] = field(default_factory=list)
    doc_by_id: dict[str, dict] = field(default_factory=dict)
    bm25: BM25Okapi | None = None
    stamp: tuple | None = None

    @property
    def count(self) -> int:
        return len(self.ids)


class Retriever:
    VECTOR_CANDIDATES = 40
    BM25_CANDIDATES = 40
    RERANK_CANDIDATES = 25
    RRF_K = 60  # standard damping constant; larger = flatter rank weighting

    def __init__(
        self,
        rerank: bool = True,
        db_dir: str | Path | None = None,
        collection_name: str = config.COLLECTION_NAME,
        collection_metadata: dict | None = None,
    ):
        # db_dir/collection_metadata are overridable so the eval harness can
        # point a retriever at a differently-built index and compare the two.
        self.db_dir = Path(db_dir or config.INDEX_DIR)
        self.collection_name = collection_name
        # None -> this project's cosine default; {} -> let Chroma pick its own
        # (squared L2), which is what the legacy index in the eval needs.
        self.collection_metadata = (
            config.COLLECTION_METADATA if collection_metadata is None
            else (collection_metadata or None)
        )
        self.client = chromadb.PersistentClient(path=str(self.db_dir))
        self.embed_model = get_sentence_transformer(config.EMBED_MODEL)
        self.rerank_enabled = rerank
        self._cross_encoder = None
        self._lock = threading.Lock()
        self._snap = _Snapshot()
        self.refresh()

    @property
    def cross_encoder(self):
        if self._cross_encoder is None:
            self._cross_encoder = get_cross_encoder(config.CROSS_ENCODER)
        return self._cross_encoder

    # -- index ------------------------------------------------------------
    def _stamp(self, collection) -> tuple:
        """Changes whenever the index content may have: a new ingest run
        rewrites the stamp file, and a count change catches anything else."""
        try:
            mtime = (self.db_dir / STAMP_NAME).stat().st_mtime_ns
        except OSError:
            mtime = None
        return mtime, collection.count()

    def refresh(self) -> None:
        """Reopen the collection and rebuild the keyword index.

        The collection is re-fetched by name every time: a full rebuild
        deletes and recreates it, and the old handle then points at nothing.
        """
        with self._lock:
            collection = self.client.get_or_create_collection(
                self.collection_name, metadata=self.collection_metadata
            )
            data = collection.get(include=["documents", "metadatas"])
            ids = list(data.get("ids") or [])
            docs = data.get("documents") or []
            metas = data.get("metadatas") or []
            self._snap = _Snapshot(
                collection=collection,
                ids=ids,
                # One map for text *and* metadata, so a hit found only by BM25
                # still carries its source filename.
                doc_by_id={i: {"text": d, "meta": m or {}} for i, d, m in zip(ids, docs, metas)},
                bm25=BM25Okapi([tokenize(d) for d in docs]) if docs else None,
                stamp=self._stamp(collection),
            )
        log.debug("retriever refreshed: %d chunks", len(ids))

    def _ensure_fresh(self) -> _Snapshot:
        """Reload if documents were (re)indexed since the last query.

        BM25 is an in-memory structure built once; without this a running UI
        kept answering from the old documents after an ingest.
        """
        snap = self._snap
        try:
            current = self._stamp(snap.collection)
        except Exception:  # the collection was deleted under us (full rebuild)
            current = None
        if current != snap.stamp:
            self.refresh()
        return self._snap

    def stats(self) -> dict:
        """Chunk count overall and per source file."""
        snap = self._ensure_fresh()
        per_file: dict[str, int] = {}
        for record in snap.doc_by_id.values():
            src = record["meta"].get("source", "unknown")
            per_file[src] = per_file.get(src, 0) + 1
        return {"chunks": snap.count, "files": per_file}

    # -- search -----------------------------------------------------------
    def _vector_ranking(self, snap: _Snapshot, question: str, n: int) -> list[str]:
        embedding = self.embed_model.encode([question], normalize_embeddings=True).tolist()
        res = snap.collection.query(
            query_embeddings=embedding, n_results=min(n, snap.count), include=[]
        )
        return list((res.get("ids") or [[]])[0])

    @staticmethod
    def _bm25_ranking(snap: _Snapshot, question: str, n: int) -> list[str]:
        tokens = tokenize(question)
        if snap.bm25 is None or not tokens:
            return []
        scores = snap.bm25.get_scores(tokens)
        n = min(n, len(scores))
        # Partial sort: only the head is ever used.
        head = np.argpartition(-scores, n - 1)[:n]
        head = head[np.argsort(-scores[head], kind="stable")]
        return [snap.ids[i] for i in head if scores[i] > 0.0]

    def query(
        self,
        question: str,
        top_k: int = 5,
        rerank: bool | None = None,
        min_score: float | None = None,
        use_vector: bool = True,
        use_bm25: bool = True,
    ) -> list[dict]:
        """Return the `top_k` most relevant chunks, best first.

        Each result carries `score` (cross-encoder relevance when reranking is
        on, otherwise the RRF score) plus the per-arm ranks that produced it,
        which is what the eval harness reports on.

        `use_vector` / `use_bm25` exist so the eval can ablate one arm at a
        time and measure what each is actually contributing.
        """
        snap = self._ensure_fresh()
        if snap.count == 0 or not question.strip():
            return []
        use_rerank = self.rerank_enabled if rerank is None else rerank

        vector_ids = self._vector_ranking(snap, question, self.VECTOR_CANDIDATES) if use_vector else []
        bm25_ids = self._bm25_ranking(snap, question, self.BM25_CANDIDATES) if use_bm25 else []

        # --- Reciprocal Rank Fusion ---
        fused: dict[str, dict] = {}
        for arm, ranked in (("vector", vector_ids), ("bm25", bm25_ids)):
            for rank, doc_id in enumerate(ranked):
                entry = fused.setdefault(doc_id, {"rrf": 0.0, "vector_rank": None, "bm25_rank": None})
                entry["rrf"] += 1.0 / (self.RRF_K + rank + 1)
                entry[f"{arm}_rank"] = rank

        candidates = []
        for doc_id, entry in fused.items():
            record = snap.doc_by_id.get(doc_id)
            if record is None:
                continue
            candidates.append({
                "id": doc_id,
                "text": record["text"],
                "meta": record["meta"],
                "score": entry["rrf"],
                "rrf": entry["rrf"],
                "vector_rank": entry["vector_rank"],
                "bm25_rank": entry["bm25_rank"],
                "rerank_score": None,
            })
        candidates.sort(key=lambda c: c["rrf"], reverse=True)

        # --- Cross-encoder rerank ---
        if use_rerank and candidates:
            # Rerank at least as many as the caller asked for, so the returned
            # list never runs past the reranked head.
            head = candidates[: max(self.RERANK_CANDIDATES, top_k)]
            scores = self.cross_encoder.predict([[question, c["text"]] for c in head])
            for cand, score in zip(head, scores):
                cand["rerank_score"] = cand["score"] = float(score)
            head.sort(key=lambda c: c["score"], reverse=True)
            # Drop the un-reranked tail rather than appending it: its `score`
            # is an RRF value (~0.02) while these are cross-encoder logits
            # (roughly -11..+11), so mixing them makes `score` meaningless.
            candidates = head

        if min_score is not None:
            candidates = [c for c in candidates if c["score"] >= min_score]

        return candidates[:top_k]


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """The process-wide retriever. The engine and the UI share this one, so
    the index, the BM25 structure and the models are loaded once."""
    return Retriever()


if __name__ == "__main__":
    r = get_retriever()
    print(f"{r.stats()['chunks']} chunks indexed.")
    while True:
        try:
            q = input("\nTest query (empty to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        for res in r.query(q, top_k=5):
            ranks = f"v={res['vector_rank']} b={res['bm25_rank']}"
            print(f"\n[{res['score']:.3f}] {res['meta'].get('source', '?')}  ({ranks})")
            print(res["text"][:200], "...")
