"""
ingest.py
Run this whenever you add/change documents in ./documents
It embeds new/changed chunks and stores them in a local Chroma DB.
Unchanged files are skipped (hash-based) so you don't re-embed everything
every time -- this matters on CPU.

Usage:
    python ingest.py
"""

import hashlib
import json
import os
from pathlib import Path

import chromadb

from embeddings import get_sentence_transformer
from loaders import load_and_chunk_directory

PROJECT_DIR = Path(__file__).resolve().parent
DOCS_DIR = str(PROJECT_DIR / "documents")
DB_DIR = str(PROJECT_DIR / "vector_store")
HASH_CACHE_FILE = os.path.join(DB_DIR, "file_hashes.json")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small, CPU-friendly, ~80MB
COLLECTION_NAME = "nexus_documents"


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def load_hash_cache() -> dict:
    if os.path.exists(HASH_CACHE_FILE):
        with open(HASH_CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_hash_cache(cache: dict):
    os.makedirs(DB_DIR, exist_ok=True)
    with open(HASH_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def get_changed_files(docs_dir: str, cache: dict):
    changed = []
    current_files = {}
    for root, _, files in os.walk(docs_dir):
        for fname in files:
            if Path(fname).suffix.lower() not in {".txt", ".md", ".pdf", ".docx"}:
                continue
            full_path = os.path.join(root, fname)
            relative_path = Path(full_path).relative_to(Path(docs_dir).resolve()).as_posix()
            h = file_hash(full_path)
            current_files[relative_path] = h
            if cache.get(relative_path) != h:
                changed.append(relative_path)
    return changed, current_files


def main():
    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs(DB_DIR, exist_ok=True)

    cache = load_hash_cache()
    changed_files, current_files = get_changed_files(DOCS_DIR, cache)
    deleted_files = [src for src in cache if src not in current_files]

    if not changed_files and not deleted_files:
        print("No new, changed, or deleted documents. Nothing to do.")
        return

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    if deleted_files:
        print(f"Removing chunks for {len(deleted_files)} deleted file(s): {deleted_files}")
        for relative_path in deleted_files:
            existing = collection.get(where={"source": relative_path})
            if existing and existing.get("ids"):
                collection.delete(ids=existing["ids"])

    if changed_files:
        print(f"Found {len(changed_files)} new/changed file(s): {changed_files}")

        print("Loading embedding model (CPU)...")
        model = get_sentence_transformer(EMBED_MODEL_NAME)

        # Remove old chunks belonging to changed files before re-adding
        for relative_path in changed_files:
            existing = collection.get(where={"source": relative_path})
            if existing and existing.get("ids"):
                collection.delete(ids=existing["ids"])

        print("Chunking documents...")
        all_chunks = load_and_chunk_directory(DOCS_DIR)
        chunks_to_embed = [c for c in all_chunks if c["source"] in changed_files]

        if chunks_to_embed:
            print(f"Embedding {len(chunks_to_embed)} chunks...")
            texts = [c["text"] for c in chunks_to_embed]
            embeddings = model.encode(texts, show_progress_bar=True).tolist()
            collection.add(
                ids=[c["id"] for c in chunks_to_embed],
                embeddings=embeddings,
                documents=texts,
                metadatas=[
                    {"source": c["source"], "chunk_index": c["chunk_index"]}
                    for c in chunks_to_embed
                ],
            )
        else:
            print("No chunks produced from changed files (empty or unreadable).")

    save_hash_cache(current_files)
    print(f"Done. Collection now has {collection.count()} chunks total.")


if __name__ == "__main__":
    main()
