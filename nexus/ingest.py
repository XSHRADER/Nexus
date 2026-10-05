"""
ingest.py
Embeds new and changed documents into the local Chroma index.

Unchanged files are skipped by content hash, and only changed files are read
at all, so re-indexing stays fast on CPU as the document set grows. A file
that fails to load is reported and retried next run rather than being marked
as done.

The index also records *how* it was built (embedding model, chunk budget,
distance metric, chunker version). If any of that changes, the whole
collection is rebuilt automatically -- a store half-written by one chunking
scheme and half by another retrieves worse than either one alone, and nothing
about it looks broken from the outside.

Usage:
    python -m nexus.ingest              # incremental
    python -m nexus.ingest --rebuild    # force a full re-index
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import chromadb

from nexus import config
from nexus.embeddings import get_max_tokens, get_sentence_transformer
from nexus.loaders import DEFAULT_MAX_TOKENS, DEFAULT_OVERLAP_TOKENS, LOADERS, chunk_file
from nexus.retrieve import STAMP_NAME

SUPPORTED = frozenset(LOADERS)
# Chroma rejects very large single add() calls.
ADD_BATCH = 256


def index_config() -> dict:
    """Everything that, if changed, invalidates the existing vectors."""
    return {
        "embed_model": config.EMBED_MODEL,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "overlap_tokens": DEFAULT_OVERLAP_TOKENS,
        "space": config.COLLECTION_METADATA["hnsw:space"],
        # v3: DOCX tables are read, text files lose their BOM.
        "chunker": "token-aware-v3",
    }


@dataclass
class Report:
    """What one ingest run did, for the CLI and the UI."""

    rebuilt: bool = False
    rebuild_reason: str | None = None
    added: list[str] = field(default_factory=list)      # new or changed, indexed
    removed: list[str] = field(default_factory=list)    # deleted from documents/
    failed: dict[str, str] = field(default_factory=dict)  # file -> error
    chunks_added: int = 0
    total_chunks: int = 0
    total_files: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.rebuilt or self.added or self.removed)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _cache_path(index_dir: Path) -> Path:
    return index_dir / STAMP_NAME


def load_cache(index_dir: Path) -> tuple[dict, dict]:
    """Returns (config, {relative_path: hash})."""
    try:
        data = json.loads(_cache_path(index_dir).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, {}
    if isinstance(data, dict) and "files" in data:
        return data.get("config", {}), data.get("files", {})
    # Pre-versioning cache: a flat {path: hash} map. Treat as unknown config
    # so it triggers one rebuild onto the current scheme.
    return {}, data if isinstance(data, dict) else {}


def save_cache(index_dir: Path, files: dict) -> None:
    """Also the retriever's freshness stamp: rewriting it tells a running UI
    to reload. Written via a temp file so a crash can't leave half a file."""
    index_dir.mkdir(parents=True, exist_ok=True)
    target = _cache_path(index_dir)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps({"config": index_config(), "files": files}, indent=2), encoding="utf-8")
    tmp.replace(target)


def scan(docs_dir: Path) -> dict[str, Path]:
    """{relative posix path: absolute path} for every supported file."""
    found = {}
    for path in sorted(docs_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED and not path.name.startswith("~$"):
            found[path.relative_to(docs_dir).as_posix()] = path
    return found


def run(
    force_rebuild: bool = False,
    docs_dir: Path | None = None,
    index_dir: Path | None = None,
    echo: Callable[[str], None] = print,
) -> Report:
    """Bring the index in line with `docs_dir`. `echo` receives progress lines."""
    docs_dir = Path(docs_dir or config.DOCS_DIR)
    index_dir = Path(index_dir or config.INDEX_DIR)
    docs_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    report = Report()

    saved_config, cache = load_cache(index_dir)
    wanted = index_config()
    if force_rebuild or saved_config != wanted:
        report.rebuilt = True
        report.rebuild_reason = "rebuild requested" if force_rebuild else "index settings changed"
        if saved_config and not force_rebuild:
            for key in sorted(set(saved_config) | set(wanted)):
                if saved_config.get(key) != wanted.get(key):
                    echo(f"  {key}: {saved_config.get(key)!r} -> {wanted.get(key)!r}")
        cache = {}

    client = chromadb.PersistentClient(path=str(index_dir))
    if report.rebuilt:
        echo(f"Full rebuild ({report.rebuild_reason}).")
        # Through Chroma rather than deleting files: the distance metric is
        # fixed when a collection is created, and a running UI may hold the
        # store's files open.
        try:
            client.delete_collection(config.COLLECTION_NAME)
        except Exception:  # absent on first run; any other error resurfaces below
            pass
    collection = client.get_or_create_collection(
        config.COLLECTION_NAME, metadata=config.COLLECTION_METADATA
    )

    files = scan(docs_dir)
    hashes = {rel: file_hash(path) for rel, path in files.items()}
    changed = [rel for rel, h in hashes.items() if cache.get(rel) != h]
    report.removed = [rel for rel in cache if rel not in files]

    for rel in report.removed + changed:
        stale = collection.get(where={"source": rel}, include=[])
        if stale.get("ids"):
            collection.delete(ids=stale["ids"])
    if report.removed:
        echo(f"Removed {len(report.removed)} deleted file(s): {', '.join(report.removed)}")

    chunks: list[dict] = []
    if changed:
        echo(f"Reading {len(changed)} new or changed file(s)...")
        for rel in changed:
            try:
                file_chunks = chunk_file(files[rel], rel)
            except Exception as exc:  # one bad file must not stop the rest
                report.failed[rel] = str(exc)
                echo(f"  [skip] {rel}: {exc}")
                continue
            if not file_chunks:
                echo(f"  [empty] {rel}: no text found")
            chunks.extend(file_chunks)
            report.added.append(rel)

    if chunks:
        limit = get_max_tokens()
        sizes = sorted(c["n_tokens"] for c in chunks)
        echo(f"Embedding {len(chunks)} chunks (tokens: median {sizes[len(sizes) // 2]}, "
             f"max {sizes[-1]}, embedder limit {limit})...")
        oversized = sum(1 for n in sizes if n > limit)
        if oversized:
            echo(f"  WARNING: {oversized} chunk(s) exceed the embedder limit and will be truncated.")
        model = get_sentence_transformer(config.EMBED_MODEL)
        for start in range(0, len(chunks), ADD_BATCH):
            batch = chunks[start : start + ADD_BATCH]
            texts = [c["text"] for c in batch]
            vectors = model.encode(texts, normalize_embeddings=True).tolist()
            collection.add(
                ids=[c["id"] for c in batch],
                embeddings=vectors,
                documents=texts,
                metadatas=[
                    {"source": c["source"], "chunk_index": c["chunk_index"], "n_tokens": c["n_tokens"]}
                    for c in batch
                ],
            )
        report.chunks_added = len(chunks)

    # A file that failed is left out, so the next run sees it as changed and
    # tries again (its old chunks were deleted above).
    done = {rel: h for rel, h in hashes.items() if rel not in report.failed}
    if report.changed or done != cache:
        save_cache(index_dir, done)

    report.total_chunks = collection.count()
    report.total_files = len(done)
    if not report.changed and not report.failed:
        echo(f"Index is up to date: {report.total_chunks} chunks from {report.total_files} file(s).")
    else:
        echo(f"Done. {report.total_chunks} chunks from {report.total_files} file(s)"
             + (f"; {len(report.failed)} could not be read." if report.failed else "."))
    return report


def main(force_rebuild: bool = False) -> int:
    report = run(force_rebuild=force_rebuild)
    return 1 if report.failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the NEXUS document index.")
    parser.add_argument("--rebuild", action="store_true", help="Re-create the index from scratch.")
    args = parser.parse_args()
    raise SystemExit(main(force_rebuild=args.rebuild))
