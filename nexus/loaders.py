"""
loaders.py
Reads .txt, .md, .pdf, and .docx files and splits them into overlapping
chunks ready for embedding.

Chunk sizes are measured in *embedding-model tokens*, not characters. The
embedder (all-MiniLM-L6-v2) truncates anything past 256 tokens without
raising, so character-sized chunks look fine on disk and arrive at the model
with their tails cut off -- the vector index then describes only the first
half of each chunk while BM25 still matches the whole thing.
"""

import re
from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader

# Leaves room under the 256-token embedder limit for [CLS]/[SEP] and for the
# tokenizer splitting a word differently than we estimated.
DEFAULT_MAX_TOKENS = 240
DEFAULT_OVERLAP_TOKENS = 48


def read_txt(path: str) -> str:
    # utf-8-sig drops the BOM Notepad writes, which otherwise ends up glued
    # to the first word of the first chunk.
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        return f.read()


def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text.append(page_text)
    return "\n".join(text)


def _xml_text(element) -> str:
    return "".join(node.text or "" for node in element.iter() if node.tag.endswith("}t"))


def read_docx(path: str) -> str:
    """Paragraphs and tables, in document order.

    `doc.paragraphs` alone skips every table, which in reports and specs is
    often where the facts are. Each table row becomes one `a | b | c` line.
    """
    body = DocxDocument(path).element.body
    lines: list[str] = []
    for block in body.iterchildren():
        tag = block.tag.rsplit("}", 1)[-1]
        if tag == "p":
            lines.append(_xml_text(block))
        elif tag == "tbl":
            for row in block.iter():
                if row.tag.endswith("}tr"):
                    cells = [_xml_text(c).strip() for c in row.iterchildren() if c.tag.endswith("}tc")]
                    lines.append(" | ".join(c for c in cells if c))
    return "\n".join(lines)


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


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


def _default_token_counter():
    """Count in the embedder's own tokens, or fall back to a rough estimate.

    The fallback keeps this module importable (and unit-testable) without the
    sentence-transformers stack loaded.
    """
    try:
        from nexus.embeddings import count_tokens

        count_tokens("warmup")
        return count_tokens
    except Exception:
        return lambda text: max(1, len(text) // 4)


def _atoms(text: str) -> list[str]:
    """Smallest units we are willing to keep whole: lines, then sentences."""
    out: list[str] = []
    for block in _PARAGRAPH_SPLIT.split(text):
        block = block.strip()
        if not block:
            continue
        for line in block.split("\n"):
            line = line.strip()
            if not line:
                continue
            if len(line) > 400:
                out.extend(s.strip() for s in _SENTENCE_SPLIT.split(line) if s.strip())
            else:
                out.append(line)
    return out


def _split_oversized(atom: str, count, max_tokens: int) -> list[str]:
    """Break one atom that is too big on word boundaries.

    Uses the atom's own tokens-per-word ratio so this costs one tokenizer call
    instead of one per word.
    """
    words = atom.split()
    if not words:
        return []
    total = count(atom)
    if total <= max_tokens:
        return [atom]
    per_word = total / len(words)
    step = max(1, int((max_tokens / per_word) * 0.9))
    return [" ".join(words[i : i + step]) for i in range(0, len(words), step)]


def chunk_text(
    text: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    count_tokens=None,
) -> list[str]:
    """Split `text` into chunks of at most `max_tokens` embedding tokens.

    Packs whole lines/sentences greedily up to the budget, then steps back far
    enough to carry ~`overlap_tokens` of context into the next chunk so an
    answer that straddles a boundary is still retrievable from one side.
    """
    text = (text or "").strip()
    if not text:
        return []

    count = count_tokens or _default_token_counter()

    atoms: list[str] = []
    sizes: list[int] = []
    for atom in _atoms(text):
        size = count(atom)
        if size > max_tokens:
            for piece in _split_oversized(atom, count, max_tokens):
                atoms.append(piece)
                sizes.append(count(piece))
        else:
            atoms.append(atom)
            sizes.append(size)

    if not atoms:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(atoms):
        end, total = start, 0
        while end < len(atoms) and total + sizes[end] <= max_tokens:
            total += sizes[end]
            end += 1
        if end == start:  # one atom still over budget -- emit it alone
            end = start + 1

        chunks.append("\n".join(atoms[start:end]))
        if end >= len(atoms):
            break

        # Walk back over the tail of this chunk to build the overlap.
        back, carried = end, 0
        while back > start + 1 and carried < overlap_tokens:
            back -= 1
            carried += sizes[back]
        start = back

    return chunks


def chunk_file(
    path: str | Path,
    source: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    count_tokens=None,
) -> list[dict]:
    """Load one file and return its chunks as
    {"id", "text", "source", "chunk_index", "n_tokens"}. Raises if unreadable."""
    count = count_tokens or _default_token_counter()
    chunks = chunk_text(load_document(str(path)), max_tokens, overlap_tokens, count_tokens=count)
    return [
        {
            "id": f"{source}::{i}",
            "text": chunk,
            "source": source,
            "chunk_index": i,
            "n_tokens": count(chunk),
        }
        for i, chunk in enumerate(chunks)
    ]


def load_and_chunk_directory(
    directory: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
):
    """Chunk every supported file under `directory`; unreadable files are skipped."""
    results = []
    base_dir = Path(directory).resolve()
    count = _default_token_counter()
    for path in sorted(base_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in LOADERS:
            continue
        source = path.relative_to(base_dir).as_posix()
        try:
            results.extend(chunk_file(path, source, max_tokens, overlap_tokens, count))
        except Exception as exc:
            print(f"[skip] Could not read {path}: {exc}")
    return results
