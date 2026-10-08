"""
brain.py
The background brain: what NEXUS does without being asked.

  watch      Every minute (brain.scan_seconds) it looks at documents/ and any
             extra watched folders. New or changed files in documents/ are
             indexed straight away (ingest.py already re-embeds only what
             changed), so questions about them work without pressing
             anything. Every change is logged for the digest.
  study      New and changed notes become flashcards: a local model writes
             question/answer pairs from each passage, and every answer is
             run through the truth check against the passage it came from.
             Answers the passage contradicts are thrown away; the rest are
             marked verified (supported) or unverified. Cards come back for
             review on a Leitner schedule: know it and the gap grows (1, 3,
             7, 14 days); miss it and it starts again.
  digest     Once a week (brain.digest_days): what changed in your files,
             what you asked about, how your models did, and a short summary
             of new notes written by a local model.
  inbox      Everything above lands here, newest first, until you read it.

Everything runs on this PC: generation goes through engine.local_generate,
which never uses a cloud model, whatever the cloud switch says.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from config import PROJECT_DIR, get_settings
from store import ChatStore, get_store

DOCS_DIR = PROJECT_DIR / "documents"
SUPPORTED_SUFFIXES = (".txt", ".md", ".pdf", ".docx")
LEITNER_DAYS = {1: 0.0, 2: 1.0, 3: 3.0, 4: 7.0, 5: 14.0}
RETRY_SECONDS = 10 * 60
DAY = 86400.0
MAX_PASSAGES_PER_FILE = 4

CARD_PROMPT = """Write flashcards from the passage below for someone revising it.
Use only facts the passage states. Each card has a short question and a one-sentence answer.
Write at most {n} cards; fewer is fine if the passage has less to learn.
Reply with JSON only: [{{"q": "question", "a": "answer"}}, ...]"""

DIGEST_PROMPT = """Write a short weekly digest of the notes below for their owner.
Use 3 to 6 bullet points, each starting with "- ", saying what is new or changed and why it
might matter. Use only what the notes say."""


# ---------------------------------------------------------------------------
# Watching files
# ---------------------------------------------------------------------------


def watched_folders() -> list[Path]:
    folders = [DOCS_DIR]
    for f in get_settings().watch_folders:
        path = Path(f).expanduser()
        if not path.is_absolute():
            path = PROJECT_DIR / path
        if path.is_dir() and path not in folders:
            folders.append(path)
    return folders


def scan(folders: list[Path]) -> dict[str, tuple[float, int]]:
    out: dict[str, tuple[float, int]] = {}
    for folder in folders:
        if not folder.exists():
            continue
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES and not p.name.startswith("."):
                st = p.stat()
                out[str(p.resolve())] = (round(st.st_mtime, 3), st.st_size)
    return out


def diff(old: dict[str, tuple[float, int]], new: dict[str, tuple[float, int]]) -> dict[str, list[str]]:
    return {
        "added": sorted(set(new) - set(old)),
        "changed": sorted(p for p in set(new) & set(old) if new[p] != old[p]),
        "removed": sorted(set(old) - set(new)),
    }


def _short(path: str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(PROJECT_DIR))
    except ValueError:
        return p.name


# ---------------------------------------------------------------------------
# Flashcards
# ---------------------------------------------------------------------------


def parse_cards(text: str) -> list[dict[str, str]]:
    match = re.search(r"\[.*\]", text or "", re.S)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    cards = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        q = str(item.get("q") or item.get("question") or "").strip()
        a = str(item.get("a") or item.get("answer") or "").strip()
        if len(q) > 5 and len(a) > 1:
            cards.append({"q": q, "a": a})
    return cards


def next_due(box: int, knew: bool, now: float) -> tuple[int, float]:
    """Leitner: right moves a card up a box (longer gap), wrong sends it to box 1."""
    if knew:
        box = min(5, box + 1)
        return box, now + LEITNER_DAYS[box] * DAY
    return 1, now + RETRY_SECONDS


class Brain:
    def __init__(
        self,
        store: ChatStore | None = None,
        generate: Callable[[list[dict[str, str]]], tuple[str, str]] | None = None,
        index: Callable[[], Any] | None = None,
        folders: Callable[[], list[Path]] = watched_folders,
        scorer=None,
    ):
        self._store = store
        self._generate = generate
        self._index = index
        self.folders = folders
        self.scorer = scorer
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()  # one tick at a time

    # -- plumbing ---------------------------------------------------------------

    @property
    def store(self) -> ChatStore:
        return self._store or get_store()

    def generate(self, messages):
        if self._generate is not None:
            return self._generate(messages)
        from engine import local_generate

        return local_generate(messages)

    def index(self):
        if self._index is not None:
            return self._index()
        import io
        from contextlib import redirect_stdout

        import ingest

        with redirect_stdout(io.StringIO()):
            ingest.main()
        try:  # the running app should see the new chunks
            from rag_pipeline import get_retriever

            get_retriever().refresh()
        except Exception:
            pass

    def _state(self, key: str, default: str | None = None) -> str | None:
        with self.store._connect() as db:
            row = db.execute("SELECT value FROM brain_state WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def _set_state(self, key: str, value: str) -> None:
        with self.store._connect() as db:
            db.execute("INSERT INTO brain_state (key, value) VALUES (?, ?) "
                       "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))

    # -- inbox ------------------------------------------------------------------

    def post(self, kind: str, title: str, body: str = "", data: dict | None = None) -> int:
        with self.store._connect() as db:
            cur = db.execute(
                "INSERT INTO inbox (kind, title, body, data_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (kind, title, body, json.dumps(data) if data else None, time.time()))
            return int(cur.lastrowid)

    def inbox(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.store._connect() as db:
            rows = db.execute("SELECT * FROM inbox ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        out = []
        for r in rows:
            item = dict(r)
            item["data"] = json.loads(item.pop("data_json")) if item.get("data_json") else None
            out.append(item)
        return out

    def unread(self) -> int:
        with self.store._connect() as db:
            return int(db.execute("SELECT COUNT(*) FROM inbox WHERE read_at IS NULL").fetchone()[0])

    def mark_read(self, item_id: int | None = None) -> None:
        with self.store._connect() as db:
            if item_id is None:
                db.execute("UPDATE inbox SET read_at = ? WHERE read_at IS NULL", (time.time(),))
            else:
                db.execute("UPDATE inbox SET read_at = ? WHERE id = ?", (time.time(), item_id))

    # -- watching ---------------------------------------------------------------

    def _known(self) -> dict[str, tuple[float, int]]:
        with self.store._connect() as db:
            return {r["path"]: (r["mtime"], r["size"]) for r in db.execute("SELECT * FROM file_state")}

    def _remember(self, current: dict[str, tuple[float, int]], changes: dict[str, list[str]]) -> None:
        now = time.time()
        with self.store._connect() as db:
            for path in changes["removed"]:
                db.execute("DELETE FROM file_state WHERE path = ?", (path,))
            for path in changes["added"] + changes["changed"]:
                mtime, size = current[path]
                db.execute("INSERT INTO file_state (path, mtime, size, seen_at) VALUES (?, ?, ?, ?) "
                           "ON CONFLICT(path) DO UPDATE SET mtime = excluded.mtime, "
                           "size = excluded.size, seen_at = excluded.seen_at",
                           (path, mtime, size, now))
            for change, paths in changes.items():
                for path in paths:
                    db.execute("INSERT INTO file_events (path, change, at) VALUES (?, ?, ?)",
                               (path, change, now))

    def tick(self) -> dict[str, Any]:
        """One look at the watched folders; index, make cards, and post what happened."""
        with self._lock:
            first_run = self._state("first_scan_done") is None
            current = scan(self.folders())
            changes = diff(self._known(), current)
            touched = changes["added"] + changes["changed"]
            report: dict[str, Any] = {"changes": changes, "indexed": None, "cards": 0}
            if first_run:
                # Everything looks "added" the first time; that is not news.
                self._remember(current, changes)
                self._set_state("first_scan_done", str(time.time()))
                report["first_run"] = True
                return report
            if not any(changes.values()):
                return report
            self._remember(current, changes)

            docs = str(DOCS_DIR.resolve())
            in_docs = [p for p in touched + changes["removed"] if p.startswith(docs)]
            if in_docs:
                try:
                    self.index()
                    report["indexed"] = True
                    names = ", ".join(_short(p) for p in in_docs[:5])
                    more = f" and {len(in_docs) - 5} more" if len(in_docs) > 5 else ""
                    self.post("indexed", f"Indexed {len(in_docs)} changed file(s)",
                              f"{names}{more} — questions about them now use the new text.",
                              {"files": [_short(p) for p in in_docs]})
                except Exception as exc:
                    report["indexed"] = False
                    self.post("error", "Couldn't index the changed files",
                              f"{exc.__class__.__name__}: {exc}. Questions use the old index "
                              "until `python ingest.py` succeeds.",
                              {"files": [_short(p) for p in in_docs]})
            if get_settings().study_enabled and touched:
                report["cards"] = self.make_cards(touched)
            return report

    # -- study ------------------------------------------------------------------

    def make_cards(self, paths: list[str]) -> int:
        """Flashcards from new/changed files, each answer truth-checked against
        the passage it came from."""
        import truth_check
        from loaders import load_document

        per_file = get_settings().cards_per_file
        made, discarded, kept_by_file, failures = 0, 0, {}, []
        for path in paths:
            try:
                text = load_document(path)
            except Exception as exc:
                failures.append(f"{_short(path)}: {exc}")
                continue
            passages = truth_check._split_passages(Path(path).name, text)
            passages = [p for p in passages if len(p.text.split()) >= 12][:MAX_PASSAGES_PER_FILE]
            budget = per_file
            for passage in passages:
                if budget <= 0:
                    break
                n = min(3, budget)
                try:
                    reply, _model = self.generate([
                        {"role": "system", "content": CARD_PROMPT.format(n=n)},
                        {"role": "user", "content": passage.text}])
                except Exception as exc:
                    failures.append(f"{_short(path)}: {exc}")
                    break
                for card in parse_cards(reply)[:n]:
                    report = truth_check.check(card["a"], sources=[{"source": passage.source,
                                                                    "text": passage.text}],
                                               scorer=self.scorer, passages=())
                    labels = [c["label"] for c in report["claims"]]
                    if truth_check.CONTRADICTED in labels:
                        discarded += 1  # the passage says otherwise: never study a wrong answer
                        continue
                    label = (truth_check.SUPPORTED if labels and all(l == truth_check.SUPPORTED for l in labels)
                             else truth_check.NOT_FOUND)
                    if self._save_card(_short(path), card, passage.text, label):
                        made += 1
                        budget -= 1
                        kept_by_file[_short(path)] = kept_by_file.get(_short(path), 0) + 1
        if made:
            verified = self._count_cards("check_label = 'supported'")
            self.post("cards", f"{made} new flashcard(s) from your notes",
                      "\n".join(f"- {name}: {n}" for name, n in kept_by_file.items())
                      + (f"\n\n{discarded} more {'was' if discarded == 1 else 'were'} thrown away "
                         "because their own source contradicted the answer." if discarded else "")
                      + f"\n\nOpen Study to review them. {verified} of your cards are verified "
                        "against their source.", {"files": kept_by_file, "discarded": discarded})
        elif failures:
            self.post("error", "Couldn't make flashcards", "\n".join(f"- {f}" for f in failures[:5]))
        return made

    def study_all(self) -> int:
        """Flashcards from every note already in the watched folders."""
        with self._lock:
            return self.make_cards(sorted(scan(self.folders())))

    def _save_card(self, source: str, card: dict, evidence: str, label: str) -> bool:
        now = time.time()
        with self.store._connect() as db:
            cur = db.execute(
                "INSERT OR IGNORE INTO cards (source, question, answer, evidence, check_label, "
                "box, due, created_at) VALUES (?, ?, ?, ?, ?, 1, ?, ?)",
                (source, card["q"], card["a"], evidence[:600], label, now, now))
            return cur.rowcount > 0

    def _count_cards(self, where: str = "1=1") -> int:
        with self.store._connect() as db:
            return int(db.execute(f"SELECT COUNT(*) FROM cards WHERE {where}").fetchone()[0])

    def due_cards(self, limit: int = 20, now: float | None = None) -> list[dict[str, Any]]:
        now = time.time() if now is None else now
        with self.store._connect() as db:
            rows = db.execute("SELECT * FROM cards WHERE due <= ? ORDER BY box, due LIMIT ?",
                              (now, limit)).fetchall()
        return [dict(r) for r in rows]

    def study_stats(self, now: float | None = None) -> dict[str, int]:
        now = time.time() if now is None else now
        return {"total": self._count_cards(),
                "due": self._count_cards(f"due <= {now}"),
                "verified": self._count_cards("check_label = 'supported'"),
                "mastered": self._count_cards("box >= 4")}

    def review(self, card_id: int, knew: bool, now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else now
        with self.store._connect() as db:
            row = db.execute("SELECT box FROM cards WHERE id = ?", (card_id,)).fetchone()
            if row is None:
                raise ValueError("card not found")
            box, due = next_due(int(row["box"]), knew, now)
            db.execute("UPDATE cards SET box = ?, due = ?, reviews = reviews + 1 WHERE id = ?",
                       (box, due, card_id))
        return {"card_id": card_id, "box": box, "due": due}

    # -- digest -----------------------------------------------------------------

    def digest_due(self, now: float | None = None) -> bool:
        days = get_settings().digest_days
        if days <= 0:
            return False
        now = time.time() if now is None else now
        last = float(self._state("last_digest_at", "0") or 0)
        if last == 0:  # start the clock on first run instead of a digest of nothing
            self._set_state("last_digest_at", str(now))
            return False
        return now - last >= days * DAY

    def make_digest(self, now: float | None = None) -> int:
        """What happened since the last digest, posted to the inbox."""
        now = time.time() if now is None else now
        since = float(self._state("last_digest_at", "0") or 0) or now - 7 * DAY
        with self.store._connect() as db:
            events = db.execute("SELECT path, change FROM file_events WHERE at > ? ORDER BY at",
                                (since,)).fetchall()
            questions = [r["content"] for r in db.execute(
                "SELECT content FROM messages WHERE role = 'user' AND created_at > ?", (since,))]
            tasks = Counter(json.loads(r["meta_json"]).get("task") for r in db.execute(
                "SELECT meta_json FROM messages WHERE role = 'assistant' AND created_at > ? "
                "AND meta_json IS NOT NULL", (since,)))
            ratings = db.execute("SELECT rating, COUNT(*) n FROM feedback WHERE created_at > ? "
                                 "GROUP BY rating", (since,)).fetchall()
            battles = int(db.execute("SELECT COUNT(*) FROM battles WHERE decided_at > ?",
                                     (since,)).fetchone()[0])
            reviews = int(db.execute("SELECT COALESCE(SUM(reviews), 0) FROM cards").fetchone()[0])

        latest: dict[str, str] = {}
        for e in events:
            latest[e["path"]] = e["change"]
        added = [p for p, c in latest.items() if c == "added"]
        changed = [p for p, c in latest.items() if c == "changed"]
        removed = [p for p, c in latest.items() if c == "removed"]
        days = max(1, round((now - since) / DAY))

        lines = [f"## Your week with NEXUS (last {days} day{'s' if days != 1 else ''})", ""]
        lines.append("### Files")
        if latest:
            for label, paths in (("New", added), ("Changed", changed), ("Removed", removed)):
                if paths:
                    lines.append(f"- {label}: " + ", ".join(_short(p) for p in paths[:8])
                                 + (f" and {len(paths) - 8} more" if len(paths) > 8 else ""))
        else:
            lines.append("- No changes in your watched folders.")

        summary_note = None
        notes = []
        for path in (added + changed)[:5]:
            try:
                from loaders import load_document

                notes.append(f"[{_short(path)}]\n{load_document(path)[:1500]}")
            except Exception:
                continue
        if notes:
            try:
                text, model = self.generate([{"role": "system", "content": DIGEST_PROMPT},
                                             {"role": "user", "content": "\n\n".join(notes)}])
                bullets = [l.strip() for l in text.splitlines() if l.strip().startswith(("-", "*"))]
                lines += ["", "### What's new in your notes", *(bullets or [text.strip()])]
                summary_note = f"summarised on this PC by {model}"
            except Exception as exc:
                summary_note = f"no local model could summarise the notes ({exc.__class__.__name__})"

        lines += ["", "### What you asked"]
        if questions:
            top = ", ".join(f"{t} {n}" for t, n in tasks.most_common(4) if t)
            lines.append(f"- {len(questions)} question(s)" + (f" — {top}" if top else ""))
            words = Counter(w for q in questions for w in re.findall(r"[a-z]{5,}", q.lower())
                            if w not in {"about", "which", "there", "would", "could", "should",
                                         "explain", "please", "what's", "where"})
            if words:
                lines.append("- Recurring topics: " + ", ".join(w for w, _ in words.most_common(6)))
        else:
            lines.append("- No questions this week.")
        up = sum(r["n"] for r in ratings if r["rating"] > 0)
        down = sum(r["n"] for r in ratings if r["rating"] < 0)
        if up or down or battles:
            lines.append(f"- Feedback: {up} 👍, {down} 👎, {battles} Arena vote(s)")
        stats = self.study_stats(now)
        lines += ["", "### Study",
                  f"- {stats['total']} flashcard(s), {stats['verified']} verified against their "
                  f"source, {stats['due']} due now, {stats['mastered']} mastered; "
                  f"{reviews} review(s) so far."]
        if summary_note:
            lines += ["", f"_Notes {summary_note}._"]

        self._set_state("last_digest_at", str(now))
        return self.post("digest", f"Weekly digest — {time.strftime('%d %b %Y', time.localtime(now))}",
                         "\n".join(lines), {"since": since, "files": len(latest),
                                           "questions": len(questions)})

    # -- background thread ------------------------------------------------------

    def run_once(self) -> dict[str, Any]:
        report = self.tick()
        if self.digest_due():
            report["digest"] = self.make_digest()
        return report

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def loop():
            while not self._stop.is_set():
                try:
                    self.run_once()
                except Exception as exc:  # never let the brain take the app down
                    try:
                        self.post("error", "Background brain hit an error", str(exc)[:500])
                    except Exception:
                        pass
                self._stop.wait(get_settings().brain_scan_seconds)

        self._thread = threading.Thread(target=loop, name="nexus-brain", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


_brain: Brain | None = None


def get_brain() -> Brain:
    global _brain
    if _brain is None:
        _brain = Brain()
    return _brain
