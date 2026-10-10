"""
store.py
Local SQLite storage for NEXUS: saved chats, per-answer metrics, cloud usage,
your feedback (ratings, Arena battles, corrections) and the background
brain's state (watched files, inbox, flashcards).

One file (data/nexus.db, or $NEXUS_DB), standard library only. Callers in
engine.py and the UIs wrap these calls, so a storage failure never costs an
answer.

Chats, messages and metrics have module-level functions. The other tables
belong to the modules that use them (cloud.py, feedback.py, brain.py), which
reach the same file through `get_store()`.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any

from nexus import config

# 1: chats, messages, turns. 2: cloud usage, feedback, Arena, background brain.
SCHEMA_VERSION = 2
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
CREATE TABLE IF NOT EXISTS usage (
    provider    TEXT    NOT NULL,
    day         TEXT    NOT NULL,           -- local date, YYYY-MM-DD
    requests    INTEGER NOT NULL DEFAULT 0,
    chars_in    INTEGER NOT NULL DEFAULT 0,
    chars_out   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (provider, day)
);
-- One thumbs up/down per answer (changing your mind overwrites it). The
-- question, model and task are copied in so training data survives a
-- deleted chat.
CREATE TABLE IF NOT EXISTS feedback (
    message_id  INTEGER PRIMARY KEY,
    rating      INTEGER NOT NULL,          -- +1 or -1
    reason      TEXT,
    question    TEXT,
    task        TEXT,
    model       TEXT,
    provider    TEXT,
    complexity  REAL,
    created_at  REAL    NOT NULL
);
-- Things you did that say what you wanted without a rating: asking another
-- model, correcting the task, forcing documents on or off.
CREATE TABLE IF NOT EXISTS signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT    NOT NULL,          -- regenerate | task_override | rag_override
    question    TEXT,
    task        TEXT,
    model       TEXT,
    value       TEXT,
    message_id  INTEGER,
    created_at  REAL    NOT NULL
);
-- Arena: two models answer the same prompt blind; you pick.
CREATE TABLE IF NOT EXISTS battles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id     TEXT,
    prompt      TEXT    NOT NULL,
    task        TEXT,
    complexity  REAL,
    model_a     TEXT    NOT NULL,
    provider_a  TEXT,
    answer_a    TEXT,
    model_b     TEXT    NOT NULL,
    provider_b  TEXT,
    answer_b    TEXT,
    winner      TEXT,                      -- a | b | tie | both_bad; NULL until voted
    created_at  REAL    NOT NULL,
    decided_at  REAL
);
-- Background brain. The last state seen of every watched file...
CREATE TABLE IF NOT EXISTS file_state (
    path        TEXT    PRIMARY KEY,
    mtime       REAL    NOT NULL,
    size        INTEGER NOT NULL,
    seen_at     REAL    NOT NULL
);
-- ...every change noticed, for the weekly digest...
CREATE TABLE IF NOT EXISTS file_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT    NOT NULL,
    change      TEXT    NOT NULL,          -- added | changed | removed
    at          REAL    NOT NULL
);
-- ...what NEXUS has to tell you...
CREATE TABLE IF NOT EXISTS inbox (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT    NOT NULL,          -- indexed | digest | cards | error
    title       TEXT    NOT NULL,
    body        TEXT,                      -- markdown
    data_json   TEXT,
    created_at  REAL    NOT NULL,
    read_at     REAL
);
-- ...and flashcards, scheduled Leitner-style.
CREATE TABLE IF NOT EXISTS cards (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT    NOT NULL,
    question    TEXT    NOT NULL,
    answer      TEXT    NOT NULL,
    evidence    TEXT,
    check_label TEXT,                      -- truth check: supported | not_found
    box         INTEGER NOT NULL DEFAULT 1,
    due         REAL    NOT NULL,
    reviews     INTEGER NOT NULL DEFAULT 0,
    created_at  REAL    NOT NULL,
    UNIQUE (source, question)
);
CREATE TABLE IF NOT EXISTS brain_state (
    key         TEXT    PRIMARY KEY,
    value       TEXT
);
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
    return config.db_path()


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


def list_chats(
    limit: int = 20, search: str | None = None, conn: sqlite3.Connection | None = None
) -> list[dict]:
    """Newest first. `search` matches the title or any message, case-insensitively."""
    conn = conn or _default()
    sql = "SELECT id, title, created, updated FROM chats"
    params: list[Any] = []
    if search and search.strip():
        like = "%" + search.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        sql += (" WHERE title LIKE ? ESCAPE '\\' OR id IN "
                "(SELECT chat_id FROM messages WHERE content LIKE ? ESCAPE '\\')")
        params += [like, like]
    sql += " ORDER BY updated DESC, created DESC LIMIT ?"
    params.append(limit)
    with _lock:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def get_chat(chat_id: str, conn: sqlite3.Connection | None = None) -> dict | None:
    conn = conn or _default()
    with _lock:
        row = conn.execute(
            "SELECT id, title, created, updated FROM chats WHERE id = ?", (str(chat_id),)
        ).fetchone()
    return dict(row) if row else None


def load_messages(chat_id: str, conn: sqlite3.Connection | None = None) -> list[dict]:
    """Oldest first. `id` is what ratings and later truth checks attach to."""
    conn = conn or _default()
    with _lock:
        rows = conn.execute(
            "SELECT id, role, content, meta FROM messages WHERE chat_id = ? ORDER BY id",
            (chat_id,),
        ).fetchall()
    return [
        {"id": r["id"], "role": r["role"], "content": r["content"],
         "meta": json.loads(r["meta"]) if r["meta"] else None}
        for r in rows
    ]


def get_message(message_id: int, conn: sqlite3.Connection | None = None) -> dict | None:
    conn = conn or _default()
    with _lock:
        r = conn.execute(
            "SELECT id, chat_id, role, content, meta FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
    if r is None:
        return None
    return {"id": r["id"], "chat_id": r["chat_id"], "role": r["role"], "content": r["content"],
            "meta": json.loads(r["meta"]) if r["meta"] else None}


def update_meta(
    message_id: int, updates: dict[str, Any], conn: sqlite3.Connection | None = None
) -> None:
    """Merge `updates` into a message's metadata (e.g. a later truth check)."""
    conn = conn or _default()
    with _lock:
        message = get_message(message_id, conn=conn)
        if message is None:
            return
        meta = dict(message.get("meta") or {})
        meta.update(updates)
        with conn:
            conn.execute("UPDATE messages SET meta = ? WHERE id = ?",
                         (json.dumps(meta, ensure_ascii=False, default=_jsonable), message_id))


def append_message(
    chat_id: str,
    role: str,
    content: str,
    meta: dict | None = None,
    conn: sqlite3.Connection | None = None,
) -> int:
    """Store one message; returns its id."""
    conn = conn or _default()
    now = time.time()
    meta_json = json.dumps(meta, ensure_ascii=False, default=_jsonable) if meta else None
    with _lock, conn:
        cur = conn.execute(
            "INSERT INTO messages (chat_id, role, content, meta, created) VALUES (?, ?, ?, ?, ?)",
            (chat_id, role, content, meta_json, now),
        )
        conn.execute("UPDATE chats SET updated = ? WHERE id = ?", (now, chat_id))
        return int(cur.lastrowid)


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


# -- cloud usage ---------------------------------------------------------------


def record_usage(provider: str, chars_in: int = 0, chars_out: int = 0,
                 day: str | None = None, conn: sqlite3.Connection | None = None) -> None:
    """Count one cloud request against today's total for `provider`."""
    conn = conn or _default()
    day = day or date.today().isoformat()
    with _lock, conn:
        conn.execute(
            """
            INSERT INTO usage (provider, day, requests, chars_in, chars_out)
            VALUES (?, ?, 1, ?, ?)
            ON CONFLICT(provider, day) DO UPDATE SET
                requests  = requests + 1,
                chars_in  = chars_in + excluded.chars_in,
                chars_out = chars_out + excluded.chars_out
            """,
            (provider, day, int(chars_in), int(chars_out)),
        )


def usage_today(provider: str, day: str | None = None,
                conn: sqlite3.Connection | None = None) -> int:
    conn = conn or _default()
    day = day or date.today().isoformat()
    with _lock:
        row = conn.execute(
            "SELECT requests FROM usage WHERE provider = ? AND day = ?", (provider, day)
        ).fetchone()
    return int(row["requests"]) if row else 0


def usage_for_day(day: str | None = None,
                  conn: sqlite3.Connection | None = None) -> dict[str, dict[str, int]]:
    conn = conn or _default()
    day = day or date.today().isoformat()
    with _lock:
        rows = conn.execute("SELECT * FROM usage WHERE day = ?", (day,)).fetchall()
    return {
        r["provider"]: {"requests": r["requests"], "chars_in": r["chars_in"],
                        "chars_out": r["chars_out"]}
        for r in rows
    }


# -- a handle on one database file ----------------------------------------------


class ChatStore:
    """One database file, for the modules that own their tables.

    feedback.py and brain.py run their own SQL through `_connect()`; the
    methods are the shared chat and usage calls bound to this file, so a
    `Brain(store=...)` or a test can point everything at its own database.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        connect(self.path).close()  # create the file and its tables

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        try:
            yield db
            db.commit()
        finally:
            db.close()

    def create_chat(self, title: str = "New chat") -> str:
        with self._connect() as db:
            return create_chat(title, conn=db)

    def add_message(self, chat_id: str, role: str, content: str,
                    meta: dict[str, Any] | None = None) -> int:
        with self._connect() as db:
            return append_message(chat_id, role, content or "", meta, conn=db)

    def get_messages(self, chat_id: str) -> list[dict]:
        with self._connect() as db:
            return load_messages(chat_id, conn=db)

    def get_message(self, message_id: int) -> dict | None:
        with self._connect() as db:
            return get_message(message_id, conn=db)

    def update_meta(self, message_id: int, updates: dict[str, Any]) -> None:
        with self._connect() as db:
            update_meta(message_id, updates, conn=db)

    def record_usage(self, provider: str, chars_in: int = 0, chars_out: int = 0,
                     day: str | None = None) -> None:
        with self._connect() as db:
            record_usage(provider, chars_in, chars_out, day, conn=db)

    def usage_today(self, provider: str, day: str | None = None) -> int:
        with self._connect() as db:
            return usage_today(provider, day, conn=db)

    def usage_for_day(self, day: str | None = None) -> dict[str, dict[str, int]]:
        with self._connect() as db:
            return usage_for_day(day, conn=db)


_stores: dict[str, ChatStore] = {}


def get_store() -> ChatStore:
    """The store for the configured database (follows $NEXUS_DB, like `_default`)."""
    path = str(db_path())
    with _lock:
        if path not in _stores:
            _stores[path] = ChatStore(path)
        return _stores[path]
