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
"""

import re
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

from embeddings import get_cross_encoder, get_sentence_transformer

DB_DIR = str(Path(__file__).resolve().parent / "vector_store")
COLLECTION_NAME = "nexus_documents"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Cosine, to match the normalised embeddings written by ingest.py. Chroma
# defaults to squared L2, which ranks by magnitude as well as direction.
COLLECTION_METADATA = {"hnsw:space": "cosine"}

_WORD = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word/identifier tokens.

    Splitting on non-word characters rather than whitespace means a query for
    `router` still matches `router.py`, which is most of the point of keeping
    a keyword arm next to the vector one.
    """
    return _WORD.findall(text.lower())


class Retriever:
    VECTOR_CANDIDATES = 40
    BM25_CANDIDATES = 40
    RERANK_CANDIDATES = 25
    RRF_K = 60  # standard damping constant; larger = flatter rank weighting

    def __init__(
        self,
        rerank: bool = True,
        db_dir: str | None = None,
        collection_name: str = COLLECTION_NAME,
        collection_metadata: dict | None = None,
    ):
        # db_dir/collection_metadata are overridable so eval_rag.py can point a
        # retriever at a differently-built index and compare the two.
        self.client = chromadb.PersistentClient(path=db_dir or DB_DIR)
        # None -> this project's cosine default; {} -> let Chroma pick its own
        # (squared L2), which is what the legacy index in eval_rag.py needs.
        metadata = (
            COLLECTION_METADATA if collection_metadata is None else (collection_metadata or None)
        )
        self.collection = self.client.get_or_create_collection(
            collection_name, metadata=metadata
        )
        self.embed_model = get_sentence_transformer(EMBED_MODEL_NAME)
        self.rerank_enabled = rerank

        self._cross_encoder = None
        self._bm25 = None
        self._ids: list[str] = []
        self._doc_by_id: dict[str, dict] = {}
        self._indexed_count = -1
        self._build_bm25_index()

    @property
    def cross_encoder(self):
        if self._cross_encoder is None:
            self._cross_encoder = get_cross_encoder()
        return self._cross_encoder

    # -- index ------------------------------------------------------------
    def _build_bm25_index(self) -> None:
        data = self.collection.get(include=["documents", "metadatas"])
        ids = data.get("ids") or []
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []

        self._ids = list(ids)
        # One map for text *and* metadata, so a hit found only by BM25 still
        # carries its source filename. Previously those came back with meta={}
        # and the answer cited "unknown".
        self._doc_by_id = {
            doc_id: {"text": doc, "meta": meta or {}}
            for doc_id, doc, meta in zip(ids, docs, metas)
        }
        self._indexed_count = len(ids)
        self._bm25 = BM25Okapi([tokenize(d) for d in docs]) if docs else None

    def _ensure_fresh(self) -> None:
        """Rebuild the keyword index if documents were ingested since startup.

        Chroma picks up new rows on its own, but BM25 is an in-memory
        structure built once -- without this the Streamlit process had to be
        restarted after every `python ingest.py`.
        """
        if self.collection.count() != self._indexed_count:
            self._build_bm25_index()

    def refresh(self) -> None:
        """Force a rebuild of the keyword index."""
        self._build_bm25_index()

    # -- search -----------------------------------------------------------
    def _vector_ranking(self, question: str, n: int) -> list[str]:
        embedding = self.embed_model.encode(
            [question], normalize_embeddings=True
        ).tolist()
        res = self.collection.query(
            query_embeddings=embedding,
            n_results=min(n, self._indexed_count),
            include=["documents", "metadatas"],
        )
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        # Cache anything the map somehow lacks, so ids from either arm resolve.
        for doc_id, doc, meta in zip(ids, docs, metas):
            self._doc_by_id.setdefault(doc_id, {"text": doc, "meta": meta or {}})
        return list(ids)

    def _bm25_ranking(self, question: str, n: int) -> list[str]:
        if self._bm25 is None:
            return []
        tokens = tokenize(question)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        # Rank once and keep only the head. The old code fused *every* document
        # in the corpus on every query, then did a list .index() lookup per
        # document to find its text -- O(N^2) in the size of the collection.
        ordered = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._ids[i] for i in ordered[:n] if scores[i] > 0.0]

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

        `use_vector` / `use_bm25` exist so eval_rag.py can ablate one arm at a
        time and measure what each is actually contributing.
        """
        self._ensure_fresh()
        if self._indexed_count == 0:
            return []

        use_rerank = self.rerank_enabled if rerank is None else rerank

        vector_ids = (
            self._vector_ranking(question, self.VECTOR_CANDIDATES) if use_vector else []
        )
        bm25_ids = (
            self._bm25_ranking(question, self.BM25_CANDIDATES) if use_bm25 else []
        )

        # --- Reciprocal Rank Fusion ---
        fused: dict[str, dict] = {}
        for arm, ranked in (("vector", vector_ids), ("bm25", bm25_ids)):
            for rank, doc_id in enumerate(ranked):
                entry = fused.setdefault(
                    doc_id, {"rrf": 0.0, "vector_rank": None, "bm25_rank": None}
                )
                entry["rrf"] += 1.0 / (self.RRF_K + rank + 1)
                entry[f"{arm}_rank"] = rank

        if not fused:
            return []

        candidates = []
        for doc_id, entry in fused.items():
            record = self._doc_by_id.get(doc_id)
            if record is None:
                continue
            candidates.append(
                {
                    "id": doc_id,
                    "text": record["text"],
                    "meta": record["meta"],
                    "score": entry["rrf"],
                    "rrf": entry["rrf"],
                    "vector_rank": entry["vector_rank"],
                    "bm25_rank": entry["bm25_rank"],
                    "rerank_score": None,
                }
            )
        candidates.sort(key=lambda c: c["rrf"], reverse=True)

        # --- Cross-encoder rerank ---
        if use_rerank and candidates:
            # Rerank at least as many as the caller asked for, so the returned
            # list never runs past the reranked head.
            head_n = max(self.RERANK_CANDIDATES, top_k)
            head = candidates[:head_n]
            scores = self.cross_encoder.predict([[question, c["text"]] for c in head])
            for cand, score in zip(head, scores):
                cand["rerank_score"] = float(score)
                cand["score"] = float(score)
            head.sort(key=lambda c: c["score"], reverse=True)
            # Drop the un-reranked tail rather than appending it. Its `score` is
            # still an RRF value (~0.02) while these are cross-encoder logits
            # (roughly -11..+11), so keeping both made `score` meaningless
            # across the boundary -- and any min_score threshold would have
            # admitted every tail item while rejecting genuinely reranked ones.
            candidates = head

        if min_score is not None:
            kept = [c for c in candidates if c["score"] >= min_score]
            # Never hand back nothing just because the threshold was strict;
            # the generator is told when context looks weak instead.
            candidates = kept or candidates[:1]

        return candidates[:top_k]


if __name__ == "__main__":
    r = Retriever()
    print(f"{r._indexed_count} chunks indexed.")
    q = input("Test query: ")
    for res in r.query(q, top_k=5):
        ranks = f"v={res['vector_rank']} b={res['bm25_rank']}"
        print(f"\n[{res['score']:.3f}] {res['meta'].get('source', '?')}  ({ranks})")
        print(res["text"][:200], "...")
