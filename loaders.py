"""
loaders.py
Reads .txt, .pdf, and .docx files and splits them into overlapping chunks
ready for embedding.
"""

import os
from pathlib import Path

from pypdf import PdfReader
from docx import Document as DocxDocument


def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text.append(page_text)
    return "\n".join(text)


def read_docx(path: str) -> str:
    doc = DocxDocument(path)
    return "\n".join(p.text for p in doc.paragraphs)


LOADERS = {
    ".txt": read_txt,
    ".md": read_txt,
    ".pdf": read_pdf,
    ".docx": read_docx,
}


def load_document(path: str) -> str:
    ext = Path(path).suffix.lower()
    loader = LOADERS.get(ext)
    if loader is None:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader(path)


def chunk_text(text: str, chunk_size: int = 1600, overlap: int = 240):
    """
    Recursive character chunking with no external dependencies.
    Splits on the largest natural boundary that keeps chunks under
    `chunk_size`, then stitches `overlap` characters of the previous
    chunk onto the next one so context isn't lost at the seams.
    chunk_size / overlap are in characters.
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


def load_and_chunk_directory(directory: str, chunk_size: int = 1600, overlap: int = 240):
    """
    Walks a directory, loads every supported file, and returns a list of
    dicts: {"id": ..., "text": ..., "source": ..., "chunk_index": ...}
    """
    results = []
    base_dir = Path(directory).resolve()
    for root, _, files in os.walk(base_dir):
        for fname in files:
            ext = Path(fname).suffix.lower()
            if ext not in LOADERS:
                continue
            full_path = os.path.join(root, fname)
            relative_path = Path(full_path).relative_to(base_dir).as_posix()
            try:
                raw_text = load_document(full_path)
            except Exception as e:
                print(f"[skip] Could not read {full_path}: {e}")
                continue

            chunks = chunk_text(raw_text, chunk_size, overlap)
            for i, chunk in enumerate(chunks):
                results.append(
                    {
                        "id": f"{relative_path}::{i}",
                        "text": chunk,
                        "source": relative_path,
                        "chunk_index": i,
                    }
                )
    return results
