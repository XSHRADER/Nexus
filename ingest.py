"""
ingest.py
Run this whenever you add/change documents in ./documents
It embeds new/changed chunks and stores them in a local Chroma DB.
Unchanged files are skipped (hash-based) so you don't re-embed everything
every time -- this matters on CPU.

The index also records *how* it was built (embedding model, chunk budget,
distance metric). If any of that changes, the whole collection is rebuilt
automatically -- a store half-written by one chunking scheme and half by
another retrieves worse than either one alone, and nothing about it looks
broken from the outside.

Usage:
    python ingest.py              # incremental
    python ingest.py --rebuild    # force a full re-index
"""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import chromadb

from embeddings import EMBED_MODEL_NAME, get_max_tokens, get_sentence_transformer
from loaders import DEFAULT_MAX_TOKENS, DEFAULT_OVERLAP_TOKENS, load_and_chunk_directory

PROJECT_DIR = Path(__file__).resolve().parent
DOCS_DIR = str(PROJECT_DIR / "documents")
DB_DIR = str(PROJECT_DIR / "vector_store")
HASH_CACHE_FILE = os.path.join(DB_DIR, "file_hashes.json")
COLLECTION_NAME = "nexus_documents"

# Must match retrieve.py -- normalised vectors compared by cosine.
COLLECTION_METADATA = {"hnsw:space": "cosine"}

# Chroma rejects very large single add() calls.
ADD_BATCH = 256

SUPPORTED = {".txt", ".md", ".pdf", ".docx"}


def index_config() -> dict:
    """Everything that, if changed, invalidates the existing vectors."""
    return {
        "embed_model": EMBED_MODEL_NAME,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "overlap_tokens": DEFAULT_OVERLAP_TOKENS,
        "space": COLLECTION_METADATA["hnsw:space"],
        "chunker": "token-aware-v2",
    }


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def load_cache() -> tuple[dict, dict]:
    """Returns (config, {relative_path: hash})."""
    if not os.path.exists(HASH_CACHE_FILE):
        return {}, {}
    try:
        with open(HASH_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}, {}
    if isinstance(data, dict) and "files" in data:
        return data.get("config", {}), data.get("files", {})
    # Pre-versioning cache: a flat {path: hash} map. Treat as unknown config
    # so it triggers one rebuild onto the current scheme.
    return {}, data if isinstance(data, dict) else {}


def save_cache(files: dict) -> None:
    os.makedirs(DB_DIR, exist_ok=True)
    with open(HASH_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump({"config": index_config(), "files": files}, f, indent=2)


def scan(docs_dir: str, cache: dict) -> tuple[list[str], dict]:
    changed, current = [], {}
    for root, _, files in os.walk(docs_dir):
        for fname in files:
            if Path(fname).suffix.lower() not in SUPPORTED:
                continue
            full_path = os.path.join(root, fname)
            rel = Path(full_path).relative_to(Path(docs_dir).resolve()).as_posix()
            h = file_hash(full_path)
            current[rel] = h
            if cache.get(rel) != h:
                changed.append(rel)
    return changed, current


def main(force_rebuild: bool = False) -> int:
    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs(DB_DIR, exist_ok=True)

    saved_config, cache = load_cache()
    config_changed = saved_config != index_config()
    rebuild = force_rebuild or config_changed

    if rebuild and saved_config:
        reason = "--rebuild requested" if force_rebuild else "index config changed"
        print(f"Full rebuild ({reason}).")
        if config_changed and saved_config:
            for key in set(saved_config) | set(index_config()):
                was, now = saved_config.get(key), index_config().get(key)
                if was != now:
                    print(f"  {key}: {was!r} -> {now!r}")

    if rebuild:
        cache = {}
        if os.path.isdir(DB_DIR):
            # Drop the whole store: the distance metric is fixed at collection
            # creation, so it cannot be changed in place.
            for entry in Path(DB_DIR).iterdir():
                if entry.name == "file_hashes.json":
                    continue
                shutil.rmtree(entry) if entry.is_dir() else entry.unlink()

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(
        COLLECTION_NAME, metadata=COLLECTION_METADATA
    )

    changed_files, current_files = scan(DOCS_DIR, cache)
    deleted_files = [src for src in cache if src not in current_files]

    if not changed_files and not deleted_files:
        print(f"No new, changed, or deleted documents. {collection.count()} chunks indexed.")
        return 0

    if deleted_files:
        print(f"Removing chunks for {len(deleted_files)} deleted file(s): {deleted_files}")
        for rel in deleted_files:
            existing = collection.get(where={"source": rel})
            if existing and existing.get("ids"):
                collection.delete(ids=existing["ids"])

    if changed_files:
        print(f"Found {len(changed_files)} new/changed file(s): {changed_files}")
        print("Loading embedding model (CPU)...")
        model = get_sentence_transformer(EMBED_MODEL_NAME)
        limit = get_max_tokens()

        for rel in changed_files:
            existing = collection.get(where={"source": rel})
            if existing and existing.get("ids"):
                collection.delete(ids=existing["ids"])

        print("Chunking documents...")
        all_chunks = load_and_chunk_directory(DOCS_DIR)
        to_embed = [c for c in all_chunks if c["source"] in changed_files]

        if not to_embed:
            print("No chunks produced from changed files (empty or unreadable).")
        else:
            oversized = [c for c in to_embed if c["n_tokens"] > limit]
            if oversized:
                # Should be impossible now that chunking is token-aware; if it
                # ever fires, the vectors would silently describe only the
                # first `limit` tokens of those chunks.
                print(
                    f"  WARNING: {len(oversized)} chunk(s) exceed the {limit}-token "
                    f"embedder limit and will be truncated."
                )
            token_counts = [c["n_tokens"] for c in to_embed]
            print(
                f"Embedding {len(to_embed)} chunks "
                f"(tokens: min={min(token_counts)} "
                f"median={sorted(token_counts)[len(token_counts) // 2]} "
                f"max={max(token_counts)}, limit={limit})..."
            )

            for start in range(0, len(to_embed), ADD_BATCH):
                batch = to_embed[start : start + ADD_BATCH]
                texts = [c["text"] for c in batch]
                embeddings = model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=len(to_embed) > ADD_BATCH,
                ).tolist()
                collection.add(
                    ids=[c["id"] for c in batch],
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=[
                        {
                            "source": c["source"],
                            "chunk_index": c["chunk_index"],
                            "n_tokens": c["n_tokens"],
                        }
                        for c in batch
                    ],
                )

    save_cache(current_files)
    total = collection.count()
    print(f"Done. Collection now has {total} chunks from {len(current_files)} file(s).")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the NEXUS document index.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and re-create the index from scratch.",
    )
    args = parser.parse_args()
    raise SystemExit(main(force_rebuild=args.rebuild))
