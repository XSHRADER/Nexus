"""
config.py
Every setting NEXUS reads, in one place.

Each value has a sensible default and can be overridden with an environment
variable, so nothing here needs editing to run on another machine. Values are
read once, at import; tests set the variables before importing NEXUS.

    NEXUS_OLLAMA_URL   Ollama base URL (default: OLLAMA_HOST, else localhost:11434)
    NEXUS_NUM_CTX      context window requested from Ollama, in tokens (8192)
    NEXUS_DATA_DIR     saved chats, metrics and logs (./data)
    NEXUS_DB           SQLite file (<data dir>/nexus.db)
    NEXUS_DOCS_DIR     documents to index (./documents)
    NEXUS_INDEX_DIR    vector index (./vector_store)
    NEXUS_LOG_LEVEL    DEBUG | INFO | WARNING (INFO)
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent


def _env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def _path(name: str, default: Path) -> Path:
    value = _env(name)
    return Path(value).expanduser().resolve() if value else default


def _int(name: str, default: int, minimum: int) -> int:
    value = _env(name)
    if value is None:
        return default
    try:
        return max(minimum, int(value))
    except ValueError:
        return default


def _ollama_url() -> str:
    """NEXUS_OLLAMA_URL, else Ollama's own OLLAMA_HOST, else the default port.

    OLLAMA_HOST is the address the *server* binds to, so it is often
    `0.0.0.0` or a bare `host:port`; a client can't connect to 0.0.0.0.
    """
    raw = _env("NEXUS_OLLAMA_URL") or _env("OLLAMA_HOST") or "http://127.0.0.1:11434"
    if "://" not in raw:
        raw = f"http://{raw}"
    scheme, _, rest = raw.partition("://")
    host, _, tail = rest.partition("/")
    if host.startswith("0.0.0.0"):
        host = "127.0.0.1" + host[len("0.0.0.0"):]
    if ":" not in host.rsplit("]", 1)[-1]:
        host += ":11434"
    return f"{scheme}://{host}" + (f"/{tail}".rstrip("/") if tail else "")


# -- paths --------------------------------------------------------------------
DOCS_DIR = _path("NEXUS_DOCS_DIR", PROJECT_DIR / "documents")
INDEX_DIR = _path("NEXUS_INDEX_DIR", PROJECT_DIR / "vector_store")
DATA_DIR = _path("NEXUS_DATA_DIR", PROJECT_DIR / "data")
ROUTER_LOG = DATA_DIR / "router_logs.jsonl"
APP_LOG = DATA_DIR / "nexus.log"
GOLDEN_SET = PROJECT_DIR / "eval" / "golden_set.json"


def db_path() -> Path:
    """Read on every call: the test suites point each module at its own file."""
    return _path("NEXUS_DB", DATA_DIR / "nexus.db")


# -- Ollama -------------------------------------------------------------------
OLLAMA_URL = _ollama_url()
# Ollama's default window is 2k-4k tokens depending on version, and it silently
# drops the front of anything longer. Always ask for this much explicitly.
NUM_CTX = _int("NEXUS_NUM_CTX", 8192, minimum=2048)
# Suggested by the launcher when nothing is installed.
DEFAULT_PULL_MODEL = "llama3.1:8b"

# -- retrieval ----------------------------------------------------------------
EMBED_MODEL = "all-MiniLM-L6-v2"
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION_NAME = "nexus_documents"
# Cosine over normalised vectors. Chroma defaults to squared L2, which ranks by
# magnitude as well as direction.
COLLECTION_METADATA = {"hnsw:space": "cosine"}
# Chunks are <=240 tokens, so 10 is ~2,200 tokens of context. eval_rag measured
# grounded@5 at 0.944 with top_k=5 and 1.000 with top_k=10.
DEFAULT_TOP_K = 10

LOG_LEVEL = (_env("NEXUS_LOG_LEVEL") or "INFO").upper()
