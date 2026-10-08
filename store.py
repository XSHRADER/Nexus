"""
store.py
Saved conversations, cloud usage and your feedback (ratings, Arena battles,
corrections), in one local SQLite file (`nexus.db` by default).

Chats used to live only in the browser session, so a refresh lost them. Each
chat is now a row in `chats` and each turn a row in `messages`, including the
metadata the UI shows under an answer (model, sources, routing) so a reopened
chat looks the way it did.

Standard library only. A connection is opened per call: SQLite handles that
cheaply, and it keeps the store safe to use from Streamlit's script reruns and
from the threaded HTTP server alike.
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import date
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 4

SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT    NOT NULL,
    created_at  REAL    NOT NULL,
    updated_at  REAL    NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id     INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role        TEXT    NOT NULL,
    content     TEXT    NOT NULL,
    meta_json   TEXT,
    created_at  REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, id);
CREATE TABLE IF NOT EXISTS usage (
    provider    TEXT    NOT NULL,
    day         TEXT    NOT NULL,           -- local date, YYYY-MM-DD
    requests    INTEGER NOT NULL DEFAULT 0,
    chars_in    INTEGER NOT NULL DEFAULT 0,
    chars_out   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (provider, day)
);
-- One 👍/👎 per answer (changing your mind overwrites it). The question,
-- model and task are copied in so training data survives a deleted chat.
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
    chat_id     INTEGER,
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
-- Background brain (Phase 6). The last state seen of every watched file...
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

TITLE_CHARS = 60


def make_title(text: str) -> str:
    """First line of the opening question, cut at a word boundary."""
    line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    if not line:
        return "New chat"
    if len(line) <= TITLE_CHARS:
        return line
    cut = line[:TITLE_CHARS].rsplit(" ", 1)[0] or line[:TITLE_CHARS]
    return cut + "…"


class ChatStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(SCHEMA)
            db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

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

    # -- chats -----------------------------------------------------------------

    def create_chat(self, title: str = "New chat") -> int:
        now = time.time()
        with self._connect() as db:
            cur = db.execute(
                "INSERT INTO chats (title, created_at, updated_at) VALUES (?, ?, ?)",
                (make_title(title), now, now),
            )
            return int(cur.lastrowid)

    def list_chats(self, limit: int = 50) -> list[dict[str, Any]]:
        """Most recently active first, with a message count."""
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT c.id, c.title, c.created_at, c.updated_at,
                       COUNT(m.id) AS message_count
                FROM chats c LEFT JOIN messages m ON m.chat_id = c.id
                GROUP BY c.id
                ORDER BY c.updated_at DESC, c.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_chat(self, chat_id: int) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
        return dict(row) if row else None

    def rename_chat(self, chat_id: int, title: str) -> None:
        with self._connect() as db:
            db.execute("UPDATE chats SET title = ? WHERE id = ?", (make_title(title), chat_id))

    def delete_chat(self, chat_id: int) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM chats WHERE id = ?", (chat_id,))

    # -- messages --------------------------------------------------------------

    def add_message(
        self,
        chat_id: int,
        role: str,
        content: str,
        meta: dict[str, Any] | None = None,
    ) -> int:
        now = time.time()
        meta_json = json.dumps(meta, ensure_ascii=False, default=str) if meta else None
        with self._connect() as db:
            cur = db.execute(
                "INSERT INTO messages (chat_id, role, content, meta_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (chat_id, role, content or "", meta_json, now),
            )
            db.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (now, chat_id))
            return int(cur.lastrowid)

    def get_messages(self, chat_id: int) -> list[dict[str, Any]]:
        """Oldest first, as `{id, role, content, meta?}` -- the shape the UIs use."""
        with self._connect() as db:
            rows = db.execute(
                "SELECT id, role, content, meta_json FROM messages "
                "WHERE chat_id = ? ORDER BY id",
                (chat_id,),
            ).fetchall()
        out = []
        for r in rows:
            message: dict[str, Any] = {"id": r["id"], "role": r["role"], "content": r["content"]}
            if r["meta_json"]:
                try:
                    message["meta"] = json.loads(r["meta_json"])
                except json.JSONDecodeError:
                    pass
            out.append(message)
        return out


    def get_message(self, message_id: int) -> dict[str, Any] | None:
        with self._connect() as db:
            r = db.execute(
                "SELECT id, chat_id, role, content, meta_json FROM messages WHERE id = ?",
                (message_id,),
            ).fetchone()
        if r is None:
            return None
        message: dict[str, Any] = {"id": r["id"], "chat_id": r["chat_id"], "role": r["role"],
                                   "content": r["content"]}
        if r["meta_json"]:
            try:
                message["meta"] = json.loads(r["meta_json"])
            except json.JSONDecodeError:
                pass
        return message

    def update_meta(self, message_id: int, updates: dict[str, Any]) -> None:
        """Merge `updates` into a message's metadata (e.g. a later truth check)."""
        message = self.get_message(message_id)
        if message is None:
            return
        meta = dict(message.get("meta") or {})
        meta.update(updates)
        with self._connect() as db:
            db.execute("UPDATE messages SET meta_json = ? WHERE id = ?",
                       (json.dumps(meta, ensure_ascii=False, default=str), message_id))

    # -- cloud usage -----------------------------------------------------------

    def record_usage(self, provider: str, chars_in: int = 0, chars_out: int = 0,
                     day: str | None = None) -> None:
        """Count one cloud request against today's total for `provider`."""
        day = day or date.today().isoformat()
        with self._connect() as db:
            db.execute(
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

    def usage_today(self, provider: str, day: str | None = None) -> int:
        day = day or date.today().isoformat()
        with self._connect() as db:
            row = db.execute(
                "SELECT requests FROM usage WHERE provider = ? AND day = ?", (provider, day)
            ).fetchone()
        return int(row["requests"]) if row else 0

    def usage_for_day(self, day: str | None = None) -> dict[str, dict[str, int]]:
        day = day or date.today().isoformat()
        with self._connect() as db:
            rows = db.execute("SELECT * FROM usage WHERE day = ?", (day,)).fetchall()
        return {
            r["provider"]: {"requests": r["requests"], "chars_in": r["chars_in"],
                            "chars_out": r["chars_out"]}
            for r in rows
        }


_default: ChatStore | None = None


def get_store() -> ChatStore:
    """Process-wide store at the configured path."""
    global _default
    if _default is None:
        from config import get_settings

        _default = ChatStore(get_settings().db_path)
    return _default
