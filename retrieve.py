"""
retrieve.py
Hybrid retrieval: vector similarity (Chroma) + keyword search (BM25),
merged with simple score fusion. Hybrid beats vector-only when queries
contain exact names, error messages, code identifiers, etc.
"""

import chromadb
from rank_bm25 import BM25Okapi
from pathlib import Path

from embeddings import get_cross_encoder, get_sentence_transformer

DB_DIR = str(Path(__file__).resolve().parent / "vector_store")
COLLECTION_NAME = "nexus_documents"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


class Retriever:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=DB_DIR)
        self.collection = self.client.get_or_create_collection(COLLECTION_NAME)
        self.embed_model = get_sentence_transformer(EMBED_MODEL_NAME)
        self._cross_encoder = None
        self._bm25 = None
        self._bm25_docs = []
        self._bm25_ids = []
        self._build_bm25_index()

    @property
    def cross_encoder(self):
        if self._cross_encoder is None:
            self._cross_encoder = get_cross_encoder()
        return self._cross_encoder

    def _build_bm25_index(self):
        data = self.collection.get()
        docs = data.get("documents", [])
        ids = data.get("ids", [])
        if not docs:
            self._bm25 = None
            return
        tokenized = [d.lower().split() for d in docs]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_docs = docs
        self._bm25_ids = ids

    def refresh(self):
        """Call after re-ingesting new documents."""
        self._build_bm25_index()

    def query(self, question: str, top_k: int = 5):
        if self.collection.count() == 0:
            return []

        # --- Vector search ---
        query_embedding = self.embed_model.encode([question]).tolist()
        vector_results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k * 2, self.collection.count()),
        )
        vector_hits = {}
        for doc_id, doc, meta, dist in zip(
            vector_results["ids"][0],
            vector_results["documents"][0],
            vector_results["metadatas"][0],
            vector_results["distances"][0],
        ):
            # convert distance -> similarity score (lower distance = better)
            vector_hits[doc_id] = {
                "text": doc,
                "meta": meta,
                "vector_score": 1 / (1 + dist),
            }

        # --- BM25 keyword search ---
        bm25_scores = {}
        if self._bm25 is not None:
            tokenized_query = question.lower().split()
            scores = self._bm25.get_scores(tokenized_query)
            max_score = max(scores) if len(scores) and max(scores) > 0 else 1
            for idx, score in enumerate(scores):
                bm25_scores[self._bm25_ids[idx]] = score / max_score

        # --- Fuse scores (simple weighted sum, vector-leaning) ---
        all_ids = set(vector_hits.keys()) | set(bm25_scores.keys())
        fused = []
        for doc_id in all_ids:
            v_score = vector_hits.get(doc_id, {}).get("vector_score", 0)
            b_score = bm25_scores.get(doc_id, 0)
            combined = 0.65 * v_score + 0.35 * b_score

            if doc_id in vector_hits:
                text = vector_hits[doc_id]["text"]
                meta = vector_hits[doc_id]["meta"]
            else:
                bm25_idx = self._bm25_ids.index(doc_id)
                text = self._bm25_docs[bm25_idx]
                meta = {}

            fused.append(
                {"id": doc_id, "text": text, "meta": meta, "score": combined}
            )

        fused.sort(key=lambda x: x["score"], reverse=True)
        top_candidates = fused[:top_k * 3]

        if top_candidates:
            pairs = [[question, cand["text"]] for cand in top_candidates]
            ce_scores = self.cross_encoder.predict(pairs)
            for idx, cand in enumerate(top_candidates):
                cand["score"] = float(ce_scores[idx])
            top_candidates.sort(key=lambda x: x["score"], reverse=True)

        return top_candidates[:top_k]


if __name__ == "__main__":
    r = Retriever()
    q = input("Test query: ")
    results = r.query(q, top_k=5)
    for res in results:
        print(f"\n[{res['score']:.3f}] {res['meta'].get('source', '?')}")
        print(res["text"][:200], "...")
