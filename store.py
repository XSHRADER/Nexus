"""
store.py
Local SQLite storage for NEXUS: saved chats and per-answer metrics.

One file (data/nexus.db, or $NEXUS_DB), standard library only. Callers in
engine.py and app.py wrap these calls, so a storage failure never costs an
answer.
"""

from __future__ import annotations

import json
import os
import sqlite3
import statistics
import threading
import time
import uuid
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DB = PROJECT_DIR / "data" / "nexus.db"
SCHEMA_VERSION = 1
TITLE_CHARS = 60

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id       TEXT PRIMARY KEY,
    title    TEXT NOT NULL,
    created  REAL NOT NULL,
    updated  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id  TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role     TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content  TEXT NOT NULL,
    meta     TEXT,
    created  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_chat ON messages(chat_id, id);
CREATE TABLE IF NOT EXISTS turns (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             REAL NOT NULL,
    chat_id        TEXT,
    task           TEXT,
    model          TEXT,
    attempts       TEXT,
    rag_used       INTEGER,
    chunks_dropped INTEGER,
    ttft_ms        REAL,
    total_ms       REAL,
    load_ms        REAL,
    prompt_tokens  INTEGER,
    eval_tokens    INTEGER,
    tokens_per_s   REAL,
    truncated      INTEGER,
    stopped        INTEGER,
    error          TEXT
);
CREATE INDEX IF NOT EXISTS turns_ts ON turns(ts);
"""

TURN_COLUMNS = (
    "ts", "chat_id", "task", "model", "attempts", "rag_used", "chunks_dropped",
    "ttft_ms", "total_ms", "load_ms", "prompt_tokens", "eval_tokens",
    "tokens_per_s", "truncated", "stopped", "error",
)
_FLAGS = ("rag_used", "truncated", "stopped")
COLD_LOAD_MS = 1000.0

# One shared connection; sqlite3 objects aren't safe to use from two threads at
# once, and Streamlit runs each session on its own thread.
_lock = threading.RLock()
_conn: sqlite3.Connection | None = None
_conn_path: str | None = None


def db_path() -> Path:
    return Path(os.environ.get("NEXUS_DB") or DEFAULT_DB)


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(path) if path else db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    if conn.execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION:
        conn.executescript(_SCHEMA)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    return conn


def _default() -> sqlite3.Connection:
    """Module connection, reopened if $NEXUS_DB changes (tests switch it)."""
    global _conn, _conn_path
    path = str(db_path())
    with _lock:
        if _conn is None or _conn_path != path:
            if _conn is not None:
                _conn.close()
            _conn = connect(path)
            _conn_path = path
        return _conn


def close() -> None:
    global _conn, _conn_path
    with _lock:
        if _conn is not None:
            _conn.close()
        _conn = None
        _conn_path = None


def _jsonable(value: Any) -> Any:
    """json.dumps fallback: numpy scores become floats, anything else a string."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


# -- chats -------------------------------------------------------------------


def create_chat(title: str, conn: sqlite3.Connection | None = None) -> str:
    conn = conn or _default()
    chat_id = uuid.uuid4().hex
    now = time.time()
    title = " ".join(title.split())[:TITLE_CHARS] or "New chat"
    with _lock, conn:
        conn.execute(
            "INSERT INTO chats (id, title, created, updated) VALUES (?, ?, ?, ?)",
            (chat_id, title, now, now),
        )
    return chat_id


def list_chats(limit: int = 20, conn: sqlite3.Connection | None = None) -> list[dict]:
    conn = conn or _default()
    with _lock:
        rows = conn.execute(
            "SELECT id, title, created, updated FROM chats "
            "ORDER BY updated DESC, created DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def load_messages(chat_id: str, conn: sqlite3.Connection | None = None) -> list[dict]:
    conn = conn or _default()
    with _lock:
        rows = conn.execute(
            "SELECT role, content, meta FROM messages WHERE chat_id = ? ORDER BY id",
            (chat_id,),
        ).fetchall()
    return [
        {"role": r["role"], "content": r["content"],
         "meta": json.loads(r["meta"]) if r["meta"] else None}
        for r in rows
    ]


def append_message(
    chat_id: str,
    role: str,
    content: str,
    meta: dict | None = None,
    conn: sqlite3.Connection | None = None,
) -> None:
    conn = conn or _default()
    now = time.time()
    meta_json = json.dumps(meta, ensure_ascii=False, default=_jsonable) if meta else None
    with _lock, conn:
        conn.execute(
            "INSERT INTO messages (chat_id, role, content, meta, created) VALUES (?, ?, ?, ?, ?)",
            (chat_id, role, content, meta_json, now),
        )
        conn.execute("UPDATE chats SET updated = ? WHERE id = ?", (now, chat_id))


def delete_chat(chat_id: str, conn: sqlite3.Connection | None = None) -> None:
    conn = conn or _default()
    with _lock, conn:
        conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))


# -- turns -------------------------------------------------------------------


def record_turn(row: dict[str, Any], conn: sqlite3.Connection | None = None) -> None:
    conn = conn or _default()
    values = {k: v for k, v in row.items() if k in TURN_COLUMNS}
    values.setdefault("ts", time.time())
    if isinstance(values.get("attempts"), (list, tuple)):
        values["attempts"] = json.dumps(values["attempts"], ensure_ascii=False, default=_jsonable)
    for flag in _FLAGS:
        if values.get(flag) is not None:
            values[flag] = int(bool(values[flag]))
    columns = list(values)
    with _lock, conn:
        conn.execute(
            f"INSERT INTO turns ({', '.join(columns)}) "
            f"VALUES ({', '.join('?' for _ in columns)})",
            [values[c] for c in columns],
        )


def _turn(row: sqlite3.Row) -> dict[str, Any]:
    turn = dict(row)
    turn["attempts"] = json.loads(turn["attempts"]) if turn.get("attempts") else []
    return turn


def recent_turns(limit: int = 20, conn: sqlite3.Connection | None = None) -> list[dict]:
    conn = conn or _default()
    with _lock:
        rows = conn.execute("SELECT * FROM turns ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [_turn(r) for r in rows]


def model_stats(window: int = 500, conn: sqlite3.Connection | None = None) -> list[dict]:
    """Per-model medians and failure rate over the last `window` answers.

    Failure rate counts attempts, not answers: a model that failed and was
    replaced by the next one in the chain counts against the first model.
    """
    turns = recent_turns(window, conn=conn)
    per: dict[str, dict[str, Any]] = {}

    def bucket(model: str) -> dict[str, Any]:
        return per.setdefault(model, {
            "model": model, "answers": 0, "attempts": 0, "failures": 0,
            "cold_loads": 0, "_ms": [], "_tps": [],
        })

    for turn in turns:
        for attempt in turn["attempts"]:
            b = bucket(attempt.get("model", "?"))
            b["attempts"] += 1
            if attempt.get("error"):
                b["failures"] += 1
        if turn.get("model"):
            b = bucket(turn["model"])
            b["answers"] += 1
            if turn.get("total_ms") is not None:
                b["_ms"].append(turn["total_ms"])
            if turn.get("tokens_per_s"):
                b["_tps"].append(turn["tokens_per_s"])
            if (turn.get("load_ms") or 0) > COLD_LOAD_MS:
                b["cold_loads"] += 1

    stats = []
    for b in per.values():
        ms, tps = b.pop("_ms"), b.pop("_tps")
        b["median_ms"] = statistics.median(ms) if ms else None
        b["median_tokens_per_s"] = statistics.median(tps) if tps else None
        b["failure_rate"] = b["failures"] / b["attempts"] if b["attempts"] else 0.0
        stats.append(b)
    return sorted(stats, key=lambda b: b["answers"], reverse=True)
