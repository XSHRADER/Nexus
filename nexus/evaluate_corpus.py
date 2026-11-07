"""
evaluate_corpus.py
Retrieval measured on a library big enough for the numbers to mean something.

The demo set (eval/golden_set.json) has 18 questions over a handful of files,
so most configurations score near 100% and can't be told apart. This runs the
same pipeline on SciFact from the public BEIR benchmark: 5,183 scientific
abstracts and 300 test questions, each labelled with the abstracts that answer
it. Every abstract becomes one file, indexed by the normal ingest code, so the
indexing time is NEXUS's own.

The data (about 4 MB) is downloaded once from Hugging Face into
data/beir/scifact/ and the index is built in data/beir/scifact_index/; neither
touches your documents or your index.

Metrics are per question, judged by file: recall@k (a right abstract in the
top k), MRR, and nDCG@10, the score BEIR itself reports, so the result can be
compared with published numbers.

    python -m nexus.evaluate_corpus             # download, index, score
    python -m nexus.evaluate_corpus --limit 50  # quicker: first 50 questions
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

from nexus import config, ingest
from nexus.evaluate import CONFIGS
from nexus.retrieve import Retriever

ROOT = config.DATA_DIR / "beir" / "scifact"
INDEX = config.DATA_DIR / "beir" / "scifact_index"
RESULTS = Path(__file__).resolve().parent.parent / "eval" / "corpus_results.json"
K = 10


def download() -> tuple[Path, list[dict]]:
    """Write one file per abstract into ROOT/docs; return it and the test questions."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    def fetch(repo: str, name: str) -> str:
        return hf_hub_download(repo, name, repo_type="dataset", cache_dir=ROOT / "hf")

    docs = ROOT / "docs"
    corpus = pq.read_table(fetch("BeIR/scifact", "corpus/corpus-00000-of-00001.parquet")).to_pylist()
    if not docs.exists() or len(list(docs.iterdir())) != len(corpus):
        docs.mkdir(parents=True, exist_ok=True)
        for row in corpus:
            (docs / f"{row['_id']}.txt").write_text(f"{row['title']}\n\n{row['text']}", encoding="utf-8")

    queries = {r["_id"]: r["text"] for r in
               pq.read_table(fetch("BeIR/scifact", "queries/queries-00000-of-00001.parquet")).to_pylist()}
    relevant: dict[str, set[str]] = {}
    with open(fetch("BeIR/scifact-qrels", "test.tsv"), encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if int(row["score"]) > 0:
                relevant.setdefault(row["query-id"], set()).add(f"{row['corpus-id']}.txt")
    questions = [{"id": q, "question": queries[q], "sources": sorted(s)} for q, s in relevant.items()]
    return docs, questions


def score(retriever: Retriever, questions: list[dict], **query_kwargs) -> dict:
    """recall@1/5/10, MRR and nDCG@10 by file, plus mean seconds per question."""
    recall = {1: 0.0, 5: 0.0, 10: 0.0}
    mrr = ndcg = 0.0
    start = time.perf_counter()
    for item in questions:
        relevant = set(item["sources"])
        ranked: list[str] = []
        for r in retriever.query(item["question"], top_k=K, **query_kwargs):
            source = r["meta"].get("source")
            if source not in ranked:  # several chunks of one abstract count once
                ranked.append(source)
        hits = [i for i, s in enumerate(ranked) if s in relevant]
        for k in recall:
            recall[k] += len([i for i in hits if i < k]) / len(relevant)
        mrr += 1 / (hits[0] + 1) if hits else 0.0
        ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), K)))
        ndcg += sum(1 / math.log2(i + 2) for i in hits) / ideal
    n = len(questions)
    return {**{f"recall@{k}": round(v / n, 4) for k, v in recall.items()},
            "mrr": round(mrr / n, 4), "ndcg@10": round(ndcg / n, 4),
            "seconds_per_question": round((time.perf_counter() - start) / n, 3)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score NEXUS retrieval on the BEIR SciFact corpus.")
    parser.add_argument("--limit", type=int, help="score only the first N questions")
    parser.add_argument("--rebuild", action="store_true", help="re-index from scratch")
    args = parser.parse_args(argv)

    docs, questions = download()
    questions = questions[: args.limit] if args.limit else questions
    print(f"SciFact: {len(list(docs.iterdir()))} abstracts, {len(questions)} questions")

    start = time.perf_counter()
    report = ingest.run(force_rebuild=args.rebuild, docs_dir=docs, index_dir=INDEX, echo=lambda _l: None)
    seconds = time.perf_counter() - start
    indexing = {"files": report.total_files, "chunks": report.total_chunks,
                "files_indexed_this_run": len(report.added), "seconds": round(seconds, 1)}
    if report.added:
        indexing["files_per_second"] = round(len(report.added) / seconds, 1)
    print(f"Index: {report.total_files} files, {report.total_chunks} chunks; "
          f"{len(report.added)} indexed in {seconds:.0f} s")

    retriever = Retriever(db_dir=INDEX)
    rows = []
    for name, kwargs in CONFIGS:
        result = score(retriever, questions, **kwargs)
        rows.append({"config": name, **result})
        print(f"  {name:<24} nDCG@10 {result['ndcg@10']:.3f}  recall@10 {result['recall@10']:.3f}  "
              f"MRR {result['mrr']:.3f}  {result['seconds_per_question']:.2f} s/question")

    if not args.limit:
        RESULTS.write_text(json.dumps({"corpus": "BEIR SciFact (test)", "questions": len(questions),
                                       "indexing": indexing, "results": rows}, indent=2) + "\n",
                           encoding="utf-8")
        print(f"Saved {RESULTS.relative_to(RESULTS.parent.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
