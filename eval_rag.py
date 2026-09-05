"""
eval_rag.py
Measures retrieval quality against a fixed set of questions, so changes to
the pipeline can be judged instead of guessed at.

Judged at *file* level: a question counts as retrieved if any returned chunk
comes from one of the files that actually answers it. Chunk-level labels
would be more precise but would need re-labelling every time the chunker
changes, which defeats the purpose of using this to compare chunkers.

Metrics
    recall@k     fraction of questions with a relevant file in the top k
    MRR          mean reciprocal rank of the first relevant hit (0 if none)
    grounded@k   fraction where the assembled top-k context actually contains
                 the answer string. This is the one that tracks answer quality:
                 file-level recall saturates on a small corpus, and it is also
                 chunker-agnostic, so it stays comparable when chunk sizes
                 change underneath it.

Usage
    python eval_rag.py                # ablate the arms of the current index
    python eval_rag.py --legacy       # also rebuild + score the pre-fix index
"""

import argparse
import json
import shutil
from pathlib import Path

from retrieve import Retriever

PROJECT_DIR = Path(__file__).resolve().parent
GOLDEN_SET = PROJECT_DIR / "eval" / "golden_set.json"
DOCS_DIR = PROJECT_DIR / "documents"
LEGACY_DB = PROJECT_DIR / "vector_store_legacy"

TOP_K = 5

# Each arm combination we want a number for.
CONFIGS = [
    ("dense only (vector)", dict(use_vector=True, use_bm25=False, rerank=False)),
    ("BM25 only (keyword)", dict(use_vector=False, use_bm25=True, rerank=False)),
    ("hybrid, RRF fusion", dict(use_vector=True, use_bm25=True, rerank=False)),
    ("hybrid + cross-encoder", dict(use_vector=True, use_bm25=True, rerank=True)),
]


def load_golden() -> list[dict]:
    with open(GOLDEN_SET, "r", encoding="utf-8") as f:
        return json.load(f)["questions"]


def evaluate(retriever: Retriever, questions: list[dict], **query_kwargs) -> dict:
    """Score one configuration over the whole golden set."""
    hits = {1: 0, 3: 0, 5: 0}
    reciprocal = 0.0
    grounded = 0
    misses = []

    for item in questions:
        relevant = set(item["sources"])
        results = retriever.query(item["question"], top_k=TOP_K, **query_kwargs)
        sources = [r["meta"].get("source") for r in results]

        first = next(
            (i for i, src in enumerate(sources) if src in relevant), None
        )
        if first is not None:
            reciprocal += 1.0 / (first + 1)
            for k in hits:
                if first < k:
                    hits[k] += 1

        needle = item.get("answer_contains")
        if needle:
            context = chr(10).join(r["text"] for r in results).lower()
            if needle.lower() in context:
                grounded += 1
            else:
                misses.append(item["id"])

    n = len(questions)
    return {
        "recall@1": hits[1] / n,
        "recall@3": hits[3] / n,
        "recall@5": hits[5] / n,
        "mrr": reciprocal / n,
        "grounded@5": grounded / n,
        "misses": misses,
    }


def print_table(title: str, rows: list[tuple[str, dict]]) -> None:
    print()
    print(title)
    print("-" * 72)
    print(
        f"{'configuration':<26}{'recall@1':>10}{'recall@5':>10}"
        f"{'MRR':>8}{'grounded@5':>12}"
    )
    print("-" * 72)
    for name, m in rows:
        print(
            f"{name:<26}{m['recall@1']:>10.3f}{m['recall@5']:>10.3f}"
            f"{m['mrr']:>8.3f}{m['grounded@5']:>12.3f}"
        )
    print("-" * 72)
    for name, m in rows:
        if m["misses"]:
            print(f"  {name}: ungrounded -> {', '.join(m['misses'])}")


# ---------------------------------------------------------------------------
# Legacy index: what the pipeline produced before the chunking/metric fixes
# ---------------------------------------------------------------------------

def _legacy_chunk(text: str, chunk_size: int = 1600, overlap: int = 240) -> list[str]:
    """The original character-based chunker, kept verbatim for comparison.

    Chunks were sized in characters with no reference to the embedding
    model's 256-token limit, so most of them were silently truncated at
    embed time.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(segment: str, seps: list[str]) -> list[str]:
        if len(segment) <= chunk_size:
            return [segment]
        sep = seps[0]
        if sep == "":
            return [segment[i : i + chunk_size] for i in range(0, len(segment), chunk_size)]
        pieces = segment.split(sep)
        out: list[str] = []
        buf = ""
        for piece in pieces:
            candidate = piece if not buf else buf + sep + piece
            if len(candidate) <= chunk_size:
                buf = candidate
            else:
                if buf:
                    out.append(buf)
                if len(piece) > chunk_size:
                    out.extend(_split(piece, seps[1:]))
                    buf = ""
                else:
                    buf = piece
        if buf:
            out.append(buf)
        return out

    raw = _split(text, separators)
    if overlap <= 0 or len(raw) <= 1:
        return raw
    stitched = [raw[0]]
    for prev, cur in zip(raw, raw[1:]):
        stitched.append((prev[-overlap:] + " " + cur).strip())
    return stitched


def build_legacy_index() -> Retriever:
    """Rebuild the index exactly as it was built before the fixes."""
    import chromadb

    from embeddings import count_tokens, get_max_tokens, get_sentence_transformer
    from loaders import load_document, LOADERS

    if LEGACY_DB.exists():
        shutil.rmtree(LEGACY_DB)
    LEGACY_DB.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(LEGACY_DB))
    # No metadata -> Chroma's default squared-L2 space, as before.
    collection = client.get_or_create_collection("nexus_documents")
    model = get_sentence_transformer()
    limit = get_max_tokens()

    ids, texts, metas = [], [], []
    for path in sorted(DOCS_DIR.rglob("*")):
        if path.suffix.lower() not in LOADERS:
            continue
        rel = path.relative_to(DOCS_DIR).as_posix()
        for i, chunk in enumerate(_legacy_chunk(load_document(str(path)))):
            ids.append(f"{rel}::{i}")
            texts.append(chunk)
            metas.append({"source": rel, "chunk_index": i})

    n_tokens = [count_tokens(t) for t in texts]
    truncated = [t for t in n_tokens if t > limit]
    lost = sum(max(0, t - limit) for t in n_tokens)
    print(
        f"legacy index: {len(texts)} chunks, {len(truncated)} over the {limit}-token "
        f"limit, {lost}/{sum(n_tokens)} tokens ({100 * lost / max(1, sum(n_tokens)):.0f}%) "
        f"dropped before embedding"
    )

    # Unnormalised vectors, as before.
    embeddings = model.encode(texts).tolist()
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metas)
    return Retriever(db_dir=str(LEGACY_DB), collection_metadata={})


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate NEXUS retrieval quality.")
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Also rebuild and score the pre-fix index for comparison.",
    )
    args = parser.parse_args()

    questions = load_golden()
    print(f"Golden set: {len(questions)} questions over {DOCS_DIR.name}/")

    current = Retriever()
    print(f"Current index: {current._indexed_count} chunks")

    rows = [(name, evaluate(current, questions, **kwargs)) for name, kwargs in CONFIGS]
    print_table("CURRENT INDEX (token-aware chunks, normalised, cosine)", rows)

    if args.legacy:
        print("\nRebuilding the legacy index for comparison...")
        legacy = build_legacy_index()
        legacy_rows = [
            (name, evaluate(legacy, questions, **kwargs)) for name, kwargs in CONFIGS
        ]
        print_table("LEGACY INDEX (1600-char chunks, unnormalised, L2)", legacy_rows)

        print()
        print("legacy -> current, full pipeline (hybrid + cross-encoder):")
        for metric in ("recall@1", "recall@5", "mrr", "grounded@5"):
            print(
                f"  {metric:<12}{legacy_rows[-1][1][metric]:.3f} -> "
                f"{rows[-1][1][metric]:.3f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
