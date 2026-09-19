# Phase 1 Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the cross-site file-operation hole in `server.py`, add multi-turn chat that always fits the context window, let Auto pick any installed model, and add metrics, log rotation, timeouts and a Stop button — all tested.

**Architecture:** A new `store.py` (stdlib SQLite) holds chats and per-answer metrics. `engine.py` moves from `/api/generate` to a streamed `/api/chat` client with an explicit `num_ctx` and a pure `fit_to_window()` budget; `providers.py` learns model capabilities from Ollama's `/api/show` and infers profiles for uncatalogued models. `server.py` becomes same-origin-only with single-use `pending_id`s; `app.py` gains saved chats, history, a Reasoning panel, a Stop button and metrics views.

**Tech Stack:** Python 3.13, Streamlit 1.6x, requests, sqlite3 (stdlib), unittest; Ollama HTTP API.

**Spec:** `docs/superpowers/specs/2026-09-19-phase1-hardening-design.md`

## Global Constraints

- Python 3.13; **no new dependencies** (SQLite via stdlib `sqlite3`).
- Tests are `unittest`, run per file: `PYTHONPATH=. python tests/test_<name>.py -v`. CI runs every `tests/test_*.py`; helper modules must not match that glob.
- Every commit on branch `phase1-hardening`. **No `Co-Authored-By` trailer and no AI attribution in any commit message.**
- Constants, verbatim from the spec: `NUM_CTX = 8192` (env `NEXUS_NUM_CTX`), `REPLY_RESERVE = 1024`, `estimate_tokens = ceil(len/3)`, `OLLAMA_TIMEOUT = (5, 120)`, `TRUNCATION_RATIO = 0.98`, `FOLLOW_UP_WORDS = 12`, `DISCOVERED_SCALE = 0.9`, `MAX_BODY = 1_000_000`, `PENDING_TTL = 600.0`, log rotation at 5 MB keeping 3, DB at `data/nexus.db` (env `NEXUS_DB`), schema `user_version = 1`.
- Storage and logging failures must never fail an answer.
- Files are read/written with `encoding="utf-8"`; paths via `pathlib`.

## Refinements to the spec (decided while planning)

1. Capability flags (`tools`, `thinking`, `vision`) are read through `providers.capabilities(model)` instead of new fields on the frozen `ModelSpec`. `ModelSpec` only gains `discovered: bool`.
2. Capability tags **extend** a name-based profile rather than replacing it, so e.g. `qwen3` (thinking) keeps its general strength and `gemma3` (vision) stays a general model.
3. `ui/index.html` now HTML-escapes every message. Once same-origin is the trust boundary, model output rendered with `innerHTML` (e.g. from a hostile document in `documents/`) would be the next hole.
4. The engine retrieves via `get_retriever().query()` + `build_prompt()` directly, because the retrieval query (follow-ups) and the kept chunks (budget) can now differ from what `retrieve_context()` bundles.
5. Timeouts land with the new chat client (Task 5) instead of with log rotation, since the old generate function is deleted in Task 6.
6. Streamlit's interrupt exceptions subclass `BaseException` (verified in the installed 1.61.1 source), so they already bypass `except Exception`; `_guard` still wraps ordinary callback exceptions as the spec requires.

## File map

| file | change | responsibility |
|---|---|---|
| `server.py` | rewrite | same-origin HTTP API; `PendingStore` of single-use ids |
| `pc_agent.py` | modify | `handle()` never mutates; drops `confirm` |
| `ui/index.html` | modify | escape output; apply via `/api/apply` |
| `router.py` | modify | `DecisionLogger` rotation |
| `store.py` | create | SQLite chats/messages/turns |
| `providers.py` | modify | `capabilities()`, `discover()` |
| `engine.py` | modify | `/api/chat` client, budget, history, metrics, turn recording |
| `app.py` | modify | saved chats, history, Reasoning, Stop, Diagnostics |
| `tests/fakes.py` | create | fake Ollama/router/retriever (not a test module) |
| `tests/test_server.py`, `test_store.py`, `test_engine.py`, `test_app.py` | create | |
| `tests/test_pc_agent.py`, `test_router.py`, `test_providers.py` | extend | |
| `documents/checklist.txt`, `program_info/checklist.txt`, `README.md`, `.gitignore` | modify | |

---

### Task 1: Lock down `server.py` and remove `confirm`

**Files:**
- Rewrite: `server.py`
- Modify: `pc_agent.py:1-10` (docstring), `pc_agent.py:126-196` (`handle`)
- Modify: `engine.py:212-245` (`answer` signature + system-agent branch)
- Modify: `ui/index.html:131-212` (script)
- Create: `tests/test_server.py`
- Modify: `tests/test_pc_agent.py`

**Interfaces:**
- Produces: `pc_agent.handle(query: str, base_dir: Path) -> dict` (no `confirm`); `engine.answer(question, base_dir=None, options=None, on_token=None)` (no `confirm`); `server.PendingStore(ttl: float = 600.0, clock=time.monotonic)` with `.put(pending) -> str`, `.pop(pending_id) -> dict | None`, `.clear()`; module global `server.PENDING`; `server.MAX_BODY`; `POST /api/apply {"pending_id"}`.

- [ ] **Step 1: Write the failing pc_agent test**

In `tests/test_pc_agent.py`, change line 60 from
`preview = handle(f'sort "{root}"', root, confirm=False)` to
`preview = handle(f'sort "{root}"', root)`, and add this test to `OrganizeFlowTests`:

```python
    def test_handle_never_changes_the_disk(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "empty").mkdir()
            (root / "a.txt").write_text("x", encoding="utf-8")
            for query in (f'remove empty folders in "{root}"', f'sort "{root}"'):
                result = handle(query, root)
                self.assertTrue(result["requires_confirmation"], query)
                with self.assertRaises(TypeError):
                    handle(query, root, confirm=True)
            self.assertTrue((root / "empty").is_dir())
            self.assertTrue((root / "a.txt").exists())
```

- [ ] **Step 2: Write the failing server tests**

Create `tests/test_server.py`:

```python
import http.client
import json
import os
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

_TMP = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
os.environ["NEXUS_DB"] = os.path.join(_TMP.name, "server-test.db")

import server  # noqa: E402  (NEXUS_DB must be set before the engine loads)

PREVIEW = {
    "answer": "would move 2 files",
    "requires_confirmation": True,
    "pending": {"op": "organize", "path": "C:/somewhere"},
}


class ServerTestCase(unittest.TestCase):
    def setUp(self):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        server.PENDING.clear()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=180)
        sent = {"Content-Type": "application/json"}
        sent.update(headers or {})
        if body is not None and not isinstance(body, bytes):
            body = json.dumps(body).encode("utf-8")
        conn.request(method, path, body=body, headers=sent)
        resp = conn.getresponse()
        raw = resp.read()
        conn.close()
        try:
            payload = json.loads(raw.decode("utf-8") or "null")
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = None
        return resp.status, dict(resp.getheaders()), payload


class GuardTests(ServerTestCase):
    def test_no_cross_origin_header(self):
        status, headers, _ = self.request("GET", "/api/models")
        self.assertEqual(status, 200)
        self.assertNotIn("Access-Control-Allow-Origin", headers)

    def test_foreign_origin_is_rejected(self):
        with mock.patch.object(server, "answer") as answer:
            status, _, _ = self.request(
                "POST", "/api/chat", {"question": "hi"},
                headers={"Origin": "https://evil.example"},
            )
        self.assertEqual(status, 403)
        answer.assert_not_called()

    def test_own_origin_is_allowed(self):
        with mock.patch.object(server, "answer", return_value={"answer": "ok"}):
            status, _, payload = self.request(
                "POST", "/api/chat", {"question": "hi"},
                headers={"Origin": f"http://127.0.0.1:{self.port}"},
            )
        self.assertEqual(status, 200)
        self.assertEqual(payload["answer"], "ok")

    def test_foreign_host_is_rejected(self):
        status, _, _ = self.request("GET", "/", headers={"Host": "evil.example:8000"})
        self.assertEqual(status, 403)

    def test_text_plain_post_is_rejected(self):
        with mock.patch.object(server, "answer") as answer:
            status, _, _ = self.request(
                "POST", "/api/chat", b'{"question": "sort my downloads"}',
                headers={"Content-Type": "text/plain"},
            )
        self.assertEqual(status, 415)
        answer.assert_not_called()

    def test_oversized_body_is_rejected(self):
        status, _, _ = self.request(
            "POST", "/api/chat", b"{}",
            headers={"Content-Length": str(server.MAX_BODY + 1)},
        )
        self.assertEqual(status, 413)

    def test_malformed_json_is_rejected(self):
        status, _, _ = self.request("POST", "/api/chat", b"{not json")
        self.assertEqual(status, 400)


class PendingTests(ServerTestCase):
    def test_pending_id_replaces_raw_pending_and_works_once(self):
        with mock.patch.object(server, "answer", return_value=dict(PREVIEW)):
            status, _, payload = self.request("POST", "/api/chat", {"question": "sort it"})
        self.assertEqual(status, 200)
        self.assertNotIn("pending", payload)
        pending_id = payload["pending_id"]

        with mock.patch.object(server, "apply_pending", return_value={"answer": "done"}) as apply:
            first = self.request("POST", "/api/apply", {"pending_id": pending_id})
            second = self.request("POST", "/api/apply", {"pending_id": pending_id})
        self.assertEqual(first[0], 200)
        self.assertEqual(first[2]["answer"], "done")
        self.assertEqual(second[0], 404)
        apply.assert_called_once()
        self.assertEqual(apply.call_args.args[0], PREVIEW["pending"])

    def test_unknown_pending_id(self):
        status, _, _ = self.request("POST", "/api/apply", {"pending_id": "nope"})
        self.assertEqual(status, 404)

    def test_pending_ids_expire(self):
        now = [0.0]
        pending = server.PendingStore(ttl=10, clock=lambda: now[0])
        pending_id = pending.put({"op": "organize", "path": "x"})
        now[0] = 11.0
        self.assertIsNone(pending.pop(pending_id))


class ConfirmRegressionTests(ServerTestCase):
    def test_confirm_flag_cannot_apply_a_change(self):
        # The original exploit: a cross-site text/plain POST with confirm=true.
        # Even as a well-formed same-origin JSON request, confirm must do nothing.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "empty").mkdir()
            status, _, payload = self.request(
                "POST", "/api/chat",
                {"question": f'remove empty folders in "{root}"', "confirm": True},
            )
            self.assertEqual(status, 200)
            self.assertTrue(payload["requires_confirmation"])
            self.assertIn("pending_id", payload)
            self.assertTrue((root / "empty").is_dir())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `PYTHONPATH=. python tests/test_pc_agent.py -v` — Expected: FAIL (`test_handle_never_changes_the_disk`: no `TypeError`, empty dir removed... or preview fails).
Run: `PYTHONPATH=. python tests/test_server.py -v` — Expected: errors (`server.PENDING`/`PendingStore`/`MAX_BODY` missing, ACAO present).

- [ ] **Step 4: Make `pc_agent.handle` read-only**

Replace the module docstring's last sentences (lines 6-9) so it reads:

```python
"""
pc_agent.py
Natural-language front end for PCToolkit.

Turns "sort my downloads folder" into a concrete operation on a concrete
path. Read-only operations (analyze, find duplicates, find large files)
run immediately. Operations that change the disk (organize, delete empty
folders) return a PREVIEW plus a `pending` action. `handle()` itself never
changes the disk; only `apply(pending)` does.
"""
```

Replace `handle`'s signature and docstring:

```python
def handle(query: str, base_dir: Path) -> dict[str, Any]:
    """Route `query` to a PCToolkit operation.

    Never changes the disk: a mutating request comes back as a preview plus a
    `pending` action that only `apply()` executes.
    """
```

Replace everything from `    if intent == "empty_dirs":` to the end of `handle` (the line `    return _apply_organize(target)`) with:

```python
    if intent == "empty_dirs":
        preview = _toolkit.delete_empty_dirs(target, dry_run=True)
        if not preview["removed"]:
            return _read_only("empty_dirs", f"🧹 No empty folders under `{target}`.")
        listing = "\n".join(f"- {p}" for p in preview["removed"][:50])
        return {
            "answer": f"🧹 **{preview['removed_count']} empty folder(s)** under `{target}` "
                      f"would be removed:\n\n{listing}\n\n_Confirm to delete them._",
            "action": "empty_dirs",
            "requires_confirmation": True,
            "pending": {"op": "empty_dirs", "path": str(target)},
        }

    # intent == "organize"
    plan = _toolkit.organize_folder(target, dry_run=True)
    if not plan["planned_moves"]:
        return _read_only("organize", f"🗂️ `{target}` is already sorted — nothing to move.")
    return {
        "answer": f"🗂️ **Plan for `{target}`** ({how}) — {plan['moved_count']} file(s) "
                  f"into category folders:\n\n{_format_plan(plan['planned_moves'])}\n\n"
                  f"_Confirm to move them. This is undoable afterwards._",
        "action": "organize",
        "requires_confirmation": True,
        "pending": {"op": "organize", "path": str(target)},
    }
```

(`_apply_organize` and `apply` stay unchanged.)

- [ ] **Step 5: Remove `confirm` from `engine.answer`**

In `engine.py`, change the signature (lines 212-218) to:

```python
def answer(
    question: str,
    base_dir: Path | None = None,
    options: Options | None = None,
    on_token: Callable[[str], None] | None = None,
) -> dict[str, Any]:
```

and the system-agent call (line 238) to:

```python
        outcome = pc_agent.handle(question, base_dir)
```

- [ ] **Step 6: Rewrite `server.py`**

Replace the whole file with:

```python
"""
server.py
Dependency-free HTTP front end for NEXUS (`python run.py --server`).

It only serves its own page. Every request must be addressed to this server's
own host and port, POST bodies must be JSON, and a previewed file operation can
only be applied with the single-use `pending_id` issued alongside the preview —
so a page on another site cannot drive it.
"""

import json
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import providers
from engine import answer, apply_pending
from router import TaskRouter

PROJECT_DIR = Path(__file__).resolve().parent
UI_DIR = PROJECT_DIR / "ui"
MAX_BODY = 1_000_000   # bytes
PENDING_TTL = 600.0    # seconds a previewed action stays applicable

STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
}


class PendingStore:
    """Previewed actions, held server-side under random single-use ids.

    The browser only ever sees the id, so it can't choose what gets applied or
    where — it can only approve a preview this server produced.
    """

    def __init__(self, ttl: float = PENDING_TTL, clock: Callable[[], float] = time.monotonic):
        self._ttl = ttl
        self._clock = clock
        self._items: dict[str, tuple[dict[str, Any], float]] = {}
        self._lock = threading.Lock()

    def put(self, pending: dict[str, Any]) -> str:
        pending_id = secrets.token_urlsafe(16)
        with self._lock:
            self._purge()
            self._items[pending_id] = (pending, self._clock() + self._ttl)
        return pending_id

    def pop(self, pending_id: str) -> dict[str, Any] | None:
        with self._lock:
            self._purge()
            item = self._items.pop(pending_id, None)
        return item[0] if item else None

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def _purge(self) -> None:
        now = self._clock()
        for key in [k for k, (_, expiry) in self._items.items() if expiry <= now]:
            del self._items[key]


PENDING = PendingStore()


class Handler(BaseHTTPRequestHandler):
    # -- guards -------------------------------------------------------------

    def _allowed_hosts(self) -> set[str]:
        port = self.server.server_address[1]
        return {f"127.0.0.1:{port}", f"localhost:{port}"}

    def _same_origin(self) -> bool:
        """Reject DNS rebinding (wrong Host) and cross-site requests (wrong Origin)."""
        hosts = self._allowed_hosts()
        if self.headers.get("Host", "") not in hosts:
            self.send_json({"error": "Forbidden host"}, status=403)
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin not in {f"http://{h}" for h in hosts}:
            self.send_json({"error": "Forbidden origin"}, status=403)
            return False
        return True

    def _read_json(self) -> dict[str, Any] | None:
        """Parse a JSON object body, or send the error response and return None.

        Requiring application/json is what stops a cross-site form or
        `fetch(..., {mode: "no-cors"})`: a browser can only send this content
        type after a CORS preflight, which this server never approves.
        """
        content_type = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if content_type != "application/json":
            self.send_json({"error": "Content-Type must be application/json"}, status=415)
            return None
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_json({"error": "Bad Content-Length"}, status=400)
            return None
        if length > MAX_BODY:
            self.send_json({"error": "Request too large"}, status=413)
            return None
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_json({"error": "Malformed JSON"}, status=400)
            return None
        if not isinstance(payload, dict):
            self.send_json({"error": "Expected a JSON object"}, status=400)
            return None
        return payload

    # -- routes -------------------------------------------------------------

    def do_GET(self):
        if not self._same_origin():
            return
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.serve_file(UI_DIR / "index.html", STATIC_TYPES[".html"])
            return
        if parsed.path == "/api/models":
            self.send_json(TaskRouter.MODEL_MAP)
            return
        if parsed.path == "/api/status":
            # What NEXUS can reach right now, so an "unavailable" answer makes sense.
            self.send_json(providers.availability())
            return
        try:
            candidate = (UI_DIR / parsed.path.lstrip("/")).resolve()
            if candidate.is_file() and candidate.is_relative_to(UI_DIR):
                self.serve_file(
                    candidate,
                    STATIC_TYPES.get(candidate.suffix.lower(), "application/octet-stream"),
                )
                return
        except OSError:
            pass
        self.send_error(404, "Not found")

    def do_POST(self):
        if not self._same_origin():
            return
        path = urlparse(self.path).path
        if path not in ("/api/chat", "/api/apply"):
            self.send_error(404, "Not found")
            return
        payload = self._read_json()
        if payload is None:
            return
        if path == "/api/chat":
            self._chat(payload)
        else:
            self._apply(payload)

    def _chat(self, payload: dict[str, Any]) -> None:
        # Any "confirm" field is ignored: answering never changes the disk.
        question = str(payload.get("question", ""))
        try:
            result = answer(question, base_dir=PROJECT_DIR)
        except Exception as exc:  # pragma: no cover - UI safety path
            self.send_json({"error": str(exc)}, status=500)
            return
        pending = result.pop("pending", None)
        if pending is not None:
            result["pending_id"] = PENDING.put(pending)
        self.send_json(result)

    def _apply(self, payload: dict[str, Any]) -> None:
        pending = PENDING.pop(str(payload.get("pending_id", "")))
        if pending is None:
            self.send_json({"error": "Unknown or expired action — ask again."}, status=404)
            return
        try:
            self.send_json(apply_pending(pending, PROJECT_DIR))
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=500)

    # -- responses ----------------------------------------------------------

    def send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_file(self, path: Path, content_type: str):
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    UI_DIR.mkdir(exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
    print("NEXUS UI running at http://127.0.0.1:8000")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping NEXUS UI...")
    finally:
        server.server_close()
```

- [ ] **Step 7: Update `ui/index.html`**

In the `<script>` block, add this function directly after `const sendBtn = ...;`:

```js
      function escapeHtml(text) {
        return String(text)
          .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
          .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
      }
```

In `appendMessage`, replace
`row.innerHTML = '<div>' + text.replace(/\n/g, '<br>') + '</div>';` with
`row.innerHTML = '<div>' + escapeHtml(text).replace(/\n/g, '<br>') + '</div>';` and
`metaRow.innerHTML = meta.map(item => '<span class="badge">' + item + '</span>').join('');` with
`metaRow.innerHTML = meta.map(item => '<span class="badge">' + escapeHtml(item) + '</span>').join('');`.

Replace the whole `function send(question, confirm) { ... }` with `send` plus a new `applyPending`:

```js
      function applyPending(pendingId, button) {
        button.disabled = true;
        button.textContent = 'Applying...';
        fetch('/api/apply', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ pending_id: pendingId })
        })
        .then(async (res) => {
          const data = await res.json();
          if (!res.ok) throw new Error(data.error || 'Request failed');
          appendMessage('assistant', data.answer || 'Done.', null);
        })
        .catch((err) => appendMessage('assistant', 'Could not apply: ' + err.message, null))
        .finally(() => button.remove());
      }

      function send(question) {
        if (!question) return;
        appendMessage('user', question);
        inputEl.value = '';
        sendBtn.disabled = true;
        sendBtn.textContent = 'Thinking...';

        fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question })
        })
        .then(async (res) => {
          const data = await res.json();
          if (!res.ok) throw new Error(data.error || 'Request failed');
          // Auto-picked, never chosen by the user — show what answered and why.
          const meta = [
            'Task: ' + (data.task || 'unknown'),
            'Auto-picked: ' + (data.model || 'none') +
              (data.provider === 'ollama' ? ' (local)' : ''),
            'Difficulty: ' + Number(data.complexity || 0).toFixed(2),
            'RAG: ' + (data.needs_rag ? 'on' : 'off')
          ];
          if (data.why) meta.push(data.why);
          appendMessage('assistant', data.answer || 'No answer returned.', meta);
          if (data.info) appendMessage('assistant', 'ℹ️ ' + data.info, null);
          if (data.requires_confirmation && data.pending_id) {
            const applyBtn = document.createElement('button');
            applyBtn.textContent = 'Apply changes';
            applyBtn.style.marginTop = '10px';
            applyBtn.addEventListener('click', () => applyPending(data.pending_id, applyBtn));
            messagesEl.lastChild.appendChild(applyBtn);
            messagesEl.scrollTop = messagesEl.scrollHeight;
          }
        })
        .catch((err) => {
          appendMessage('assistant', 'NEXUS could not answer: ' + err.message, null);
        })
        .finally(() => {
          sendBtn.disabled = false;
          sendBtn.textContent = 'Send';
        });
      }
```

and change `function ask() { send(inputEl.value.trim(), false); }` to `function ask() { send(inputEl.value.trim()); }`.

- [ ] **Step 8: Run tests to verify they pass**

Run: `PYTHONPATH=. python tests/test_pc_agent.py -v` — Expected: 14 tests OK.
Run: `PYTHONPATH=. python tests/test_server.py -v` — Expected: 11 tests OK (the regression test loads the router model; allow ~30 s).

- [ ] **Step 9: Manual check of the browser UI**

Run `python server.py` in the background, open `http://127.0.0.1:8000` in the built-in browser, send `analyze "C:/Windows/Temp"`, confirm a reply renders, then stop the server.

- [ ] **Step 10: Commit**

```bash
git add server.py pc_agent.py engine.py ui/index.html tests/test_server.py tests/test_pc_agent.py
git commit -m "Close the cross-site file-operation hole in server.py

The server allowed any origin, parsed text/plain bodies as JSON and honoured
confirm:true from the request, so any web page could make it move files.
It now accepts only same-origin JSON, and a previewed action can only be
applied once through a server-issued pending_id. pc_agent.handle() no
longer mutates at all, and the browser UI escapes model output."
```

---

### Task 2: Rotate `router_logs.jsonl`

**Files:**
- Modify: `router.py:53-65` (`LOG_FILE` area and `DecisionLogger`)
- Modify: `tests/test_router.py`

**Interfaces:**
- Produces: `DecisionLogger(log_file=LOG_FILE, max_bytes=5*1024*1024, backups=3)`; constants `router.LOG_MAX_BYTES`, `router.LOG_BACKUPS`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_router.py` (add `import json`, `import tempfile`, `from pathlib import Path` to the imports and change the router import to `from router import DecisionLogger, TaskRouter`):

```python
class DecisionLoggerTests(unittest.TestCase):
    def test_rotates_and_keeps_three_backups(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "router_logs.jsonl"
            logger = DecisionLogger(log, max_bytes=200, backups=3)
            for i in range(60):
                logger.log({"i": i, "pad": "x" * 40})
            names = sorted(p.name for p in Path(d).iterdir())
            self.assertEqual(
                names,
                ["router_logs.jsonl", "router_logs.jsonl.1",
                 "router_logs.jsonl.2", "router_logs.jsonl.3"],
            )
            for path in Path(d).iterdir():
                self.assertLess(path.stat().st_size, 300)
            last = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(last["i"], 59)
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=. python tests/test_router.py -v DecisionLoggerTests` — Expected: `TypeError: ... unexpected keyword argument 'max_bytes'`.

- [ ] **Step 3: Implement rotation**

Replace the `DecisionLogger` class in `router.py` with:

```python
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUPS = 3


class DecisionLogger:
    """Appends routing decisions as JSON lines, rotating at `max_bytes`.

    router_logs.jsonl -> .1 -> .2 -> .3; the oldest falls off the end, so the
    log stays under roughly (backups + 1) * max_bytes.
    """

    def __init__(
        self,
        log_file: str | Path = LOG_FILE,
        max_bytes: int = LOG_MAX_BYTES,
        backups: int = LOG_BACKUPS,
    ):
        self.log_file = Path(log_file)
        self.max_bytes = max_bytes
        self.backups = backups
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def _backup(self, n: int) -> Path:
        return self.log_file.with_name(f"{self.log_file.name}.{n}")

    def _rotate(self) -> None:
        for n in range(self.backups, 0, -1):
            src = self.log_file if n == 1 else self._backup(n - 1)
            if src.exists():
                src.replace(self._backup(n))

    def log(self, payload: dict[str, Any]) -> None:
        try:
            if self.log_file.stat().st_size >= self.max_bytes:
                self._rotate()
        except OSError:
            # Missing file, or (on Windows) another process has it open —
            # skip rotating this time rather than lose the decision.
            pass
        with open(self.log_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=. python tests/test_router.py -v` — Expected: 26 tests OK.

- [ ] **Step 5: Commit**

```bash
git add router.py tests/test_router.py
git commit -m "Rotate router_logs.jsonl at 5 MB, keeping three backups"
```

---

### Task 3: `store.py` — chats, messages, turns

**Files:**
- Create: `store.py`
- Create: `tests/test_store.py`
- Modify: `.gitignore` (add `data/`)

**Interfaces:**
- Produces (all accept optional `conn: sqlite3.Connection | None`): `connect(path=None) -> sqlite3.Connection`, `close() -> None`, `db_path() -> Path`, `create_chat(title: str) -> str`, `list_chats(limit=20) -> list[dict]` (`id, title, created, updated`), `load_messages(chat_id) -> list[dict]` (`role, content, meta`), `append_message(chat_id, role, content, meta=None) -> None`, `delete_chat(chat_id) -> None`, `record_turn(row: dict) -> None`, `recent_turns(limit=20) -> list[dict]` (row dict, `attempts` parsed to list), `model_stats(window=500) -> list[dict]` (`model, answers, attempts, failures, failure_rate, cold_loads, median_ms, median_tokens_per_s`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_store.py`:

```python
import os
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

import store


class StoreTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.conn = store.connect(Path(self.tmp.name) / "t.db")

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()


class ChatTests(StoreTestCase):
    def test_round_trip_keeps_order_and_meta(self):
        chat = store.create_chat("  What   is\nRAG?  ", conn=self.conn)
        store.append_message(chat, "user", "What is RAG?", conn=self.conn)
        store.append_message(
            chat, "assistant", "Retrieval-augmented generation.",
            {"model": "llama3.1:8b", "sources": [{"score": np.float32(0.5)}]},
            conn=self.conn,
        )
        [row] = store.list_chats(conn=self.conn)
        self.assertEqual(row["title"], "What is RAG?")
        messages = store.load_messages(chat, conn=self.conn)
        self.assertEqual([m["role"] for m in messages], ["user", "assistant"])
        self.assertIsNone(messages[0]["meta"])
        self.assertEqual(messages[1]["meta"]["sources"][0]["score"], 0.5)

    def test_long_titles_are_trimmed(self):
        chat = store.create_chat("x" * 200, conn=self.conn)
        self.assertEqual(len(store.list_chats(conn=self.conn)[0]["title"]), 60)
        self.assertTrue(chat)

    def test_most_recently_updated_first(self):
        a = store.create_chat("a", conn=self.conn)
        time.sleep(0.02)
        b = store.create_chat("b", conn=self.conn)
        time.sleep(0.02)
        store.append_message(a, "user", "bump", conn=self.conn)
        self.assertEqual([c["id"] for c in store.list_chats(conn=self.conn)], [a, b])

    def test_delete_cascades_to_messages(self):
        chat = store.create_chat("a", conn=self.conn)
        store.append_message(chat, "user", "hello", conn=self.conn)
        store.delete_chat(chat, conn=self.conn)
        self.assertEqual(store.list_chats(conn=self.conn), [])
        self.assertEqual(store.load_messages(chat, conn=self.conn), [])


class TurnTests(StoreTestCase):
    def test_record_and_read_back(self):
        store.record_turn(
            {"task": "general", "model": "a", "attempts": [{"model": "a", "error": None}],
             "total_ms": 120.0, "tokens_per_s": 30.0, "truncated": True},
            conn=self.conn,
        )
        [turn] = store.recent_turns(conn=self.conn)
        self.assertEqual(turn["model"], "a")
        self.assertEqual(turn["attempts"], [{"model": "a", "error": None}])
        self.assertEqual(turn["truncated"], 1)
        self.assertIsNotNone(turn["ts"])

    def test_model_stats(self):
        rows = [
            {"model": "a", "attempts": [{"model": "a", "error": None}],
             "total_ms": 100.0, "tokens_per_s": 20.0, "load_ms": 50.0},
            {"model": "a", "attempts": [{"model": "a", "error": None}],
             "total_ms": 300.0, "tokens_per_s": 40.0, "load_ms": 5000.0},
            {"model": "b", "attempts": [{"model": "a", "error": "boom"},
                                        {"model": "b", "error": None}],
             "total_ms": 200.0, "tokens_per_s": 10.0, "load_ms": 10.0},
        ]
        for row in rows:
            store.record_turn(row, conn=self.conn)
        stats = {s["model"]: s for s in store.model_stats(conn=self.conn)}
        self.assertEqual(stats["a"]["answers"], 2)
        self.assertEqual(stats["a"]["attempts"], 3)
        self.assertAlmostEqual(stats["a"]["failure_rate"], 1 / 3)
        self.assertEqual(stats["a"]["median_ms"], 200.0)
        self.assertEqual(stats["a"]["median_tokens_per_s"], 30.0)
        self.assertEqual(stats["a"]["cold_loads"], 1)
        self.assertEqual(stats["b"]["failure_rate"], 0.0)


class DefaultConnectionTests(unittest.TestCase):
    def test_nexus_db_env_var_picks_the_file(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            path = Path(d) / "env.db"
            old = os.environ.get("NEXUS_DB")
            os.environ["NEXUS_DB"] = str(path)
            try:
                store.create_chat("via env")
                self.assertTrue(path.exists())
                self.assertEqual(store.list_chats()[0]["title"], "via env")
            finally:
                store.close()
                if old is None:
                    os.environ.pop("NEXUS_DB", None)
                else:
                    os.environ["NEXUS_DB"] = old


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=. python tests/test_store.py -v` — Expected: `ModuleNotFoundError: No module named 'store'`.

- [ ] **Step 3: Implement `store.py`**

```python
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
```

Add to `.gitignore` under "Generated artifacts":

```
# Saved chats and metrics (store.py)
data/
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=. python tests/test_store.py -v` — Expected: 7 tests OK.

- [ ] **Step 5: Commit**

```bash
git add store.py tests/test_store.py .gitignore
git commit -m "Add store.py: SQLite chats, messages and per-answer metrics"
```

---

### Task 4: Model capabilities and discovery in `providers.py`

**Files:**
- Modify: `providers.py` (imports; `ModelSpec`; new section after "Availability"; `plan()` loop)
- Modify: `tests/test_providers.py`

**Interfaces:**
- Produces: `providers.capabilities(model: str) -> {"caps": set[str], "context_length": int | None, "parameter_size": float | None}` (cached per process on success); `providers.discover(installed: list[str]) -> list[ModelSpec]`; `ModelSpec.discovered: bool`; `providers.DISCOVERED_SCALE = 0.9`; `providers._CAPS_CACHE` (dict, tests clear it).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_providers.py` (add `from unittest import mock` to imports):

```python
NO_CAPS = {"caps": set(), "context_length": None, "parameter_size": None}


class DiscoveryTests(unittest.TestCase):
    CAPS = {
        "qwen3:8b": {"completion", "thinking", "tools"},
        "bge-m3:latest": {"embedding"},
    }

    def setUp(self):
        patcher = mock.patch.object(
            providers, "capabilities",
            lambda m: {**NO_CAPS, "caps": self.CAPS.get(m, set())},
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _one(self, tag):
        specs = providers.discover([tag])
        self.assertEqual(len(specs), 1, specs)
        return specs[0]

    def test_unknown_general_model_gets_a_scaled_general_profile(self):
        spec = self._one("granite3.3:8b")
        self.assertTrue(spec.discovered)
        self.assertAlmostEqual(spec.strengths["general"], 0.75 * 0.9)
        self.assertEqual((spec.quality, spec.speed), (0.58, 0.80))

    def test_catalogued_models_are_not_rediscovered(self):
        self.assertEqual(providers.discover(["llama3.1:8b", "qwen2.5-coder:7b"]), [])

    def test_embedding_models_are_excluded(self):
        self.assertEqual(providers.discover(["nomic-embed-text:latest", "bge-m3:latest"]), [])

    def test_coder_family_and_size_from_the_tag(self):
        spec = self._one("qwen3-coder:30b")
        self.assertAlmostEqual(spec.strengths["coding"], 0.80 * 0.9)
        self.assertEqual((spec.quality, spec.speed), (0.75, 0.25))

    def test_reasoner_by_name(self):
        spec = self._one("qwq:32b")
        self.assertEqual(max(spec.strengths, key=spec.strengths.get), "reasoning")

    def test_thinking_capability_extends_a_general_model(self):
        spec = self._one("qwen3:8b")
        self.assertAlmostEqual(spec.strengths["general"], 0.75 * 0.9)
        self.assertAlmostEqual(spec.strengths["reasoning"], 0.78 * 0.9)

    def test_vision_by_name_is_vision_only(self):
        spec = self._one("llama3.2-vision:11b")
        self.assertEqual(set(spec.strengths), {"vision"})

    def test_catalogue_entry_beats_a_discovered_peer(self):
        chain = providers.plan(
            "general", "what is a vector database",
            avail={"ollama": ["llama3.1:8b", "granite3.3:8b"]},
        )
        self.assertEqual(chain[0].model, "llama3.1:8b")
        discovered = next(c for c in chain if c.model == "granite3.3:8b")
        self.assertIn("inferred", discovered.reason)

    def test_discovered_model_answers_when_nothing_catalogued_is_installed(self):
        chain = providers.plan("general", "hello there", avail={"ollama": ["granite3.3:8b"]})
        self.assertEqual(chain[0].model, "granite3.3:8b")


class CapabilitiesTests(unittest.TestCase):
    def setUp(self):
        providers._CAPS_CACHE.clear()
        self.addCleanup(providers._CAPS_CACHE.clear)

    def test_parses_show_and_caches(self):
        response = mock.Mock(status_code=200)
        response.json.return_value = {
            "capabilities": ["completion", "Tools"],
            "details": {"parameter_size": "8.0B"},
            "model_info": {"llama.context_length": 131072},
        }
        with mock.patch.object(providers.requests, "post", return_value=response) as post:
            info = providers.capabilities("llama3.1:8b")
            providers.capabilities("llama3.1:8b")
        self.assertEqual(info["caps"], {"completion", "tools"})
        self.assertEqual(info["context_length"], 131072)
        self.assertEqual(info["parameter_size"], 8.0)
        self.assertEqual(post.call_count, 1)

    def test_degrades_and_retries_when_ollama_is_down(self):
        with mock.patch.object(
            providers.requests, "post",
            side_effect=providers.requests.ConnectionError("down"),
        ) as post:
            info = providers.capabilities("llama3.1:8b")
            providers.capabilities("llama3.1:8b")
        self.assertEqual(info, NO_CAPS)
        self.assertEqual(post.call_count, 2)
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=. python tests/test_providers.py -v` — Expected: errors, `module 'providers' has no attribute 'discover'` / `'requests'`.

- [ ] **Step 3: Implement**

In `providers.py` add `import requests` after `from typing import Any`.

Add a field to `ModelSpec` after `note: str = ""`:

```python
    discovered: bool = False  # profile inferred by discover(), not hand-tuned
```

Add this section between the "Availability" and "Selection" sections:

```python
# ---------------------------------------------------------------------------
# Capabilities and discovery
# ---------------------------------------------------------------------------

SHOW_URL = "http://localhost:11434/api/show"

# A discovered profile is a guess, so it is scaled down: a hand-tuned catalogue
# entry wins any tie, and a discovered model answers when nothing catalogued
# fits better.
DISCOVERED_SCALE = 0.9

_GENERAL = {"general": 0.75, "coding": 0.50, "reasoning": 0.50, "planning": 0.45}
_CODER = {"coding": 0.80, "general": 0.45}
_REASONER = {"reasoning": 0.78, "planning": 0.68, "coding": 0.55, "general": 0.50}
_VISION = {"vision": 0.72}

_CAPS_CACHE: dict[str, dict[str, Any]] = {}


def _parse_size(text: Any) -> float | None:
    """'8.0B' -> 8.0, '671M' -> 0.671 (billions of parameters)."""
    match = re.match(r"^\s*([\d.]+)\s*([BbMm])", str(text or ""))
    if not match:
        return None
    value = float(match.group(1))
    return value / 1000.0 if match.group(2).lower() == "m" else value


def _size_from_tag(tag: str) -> float | None:
    match = re.search(r":(\d+(?:\.\d+)?)b\b", tag.lower())
    return float(match.group(1)) if match else None


def capabilities(model: str) -> dict[str, Any]:
    """What Ollama reports about `model`: capability tags, context window, size.

    Cached per process once Ollama answers. Returns empty values (not cached)
    when Ollama is unreachable or too old to report them, so callers degrade to
    name-only guesses and default windows.
    """
    if model in _CAPS_CACHE:
        return _CAPS_CACHE[model]
    info: dict[str, Any] = {"caps": set(), "context_length": None, "parameter_size": None}
    try:
        response = requests.post(SHOW_URL, json={"model": model}, timeout=3.0)
    except Exception:
        return info
    if response.status_code == 200:
        data = response.json()
        info["caps"] = {str(c).lower() for c in data.get("capabilities") or []}
        info["parameter_size"] = _parse_size((data.get("details") or {}).get("parameter_size"))
        for key, value in (data.get("model_info") or {}).items():
            if key.endswith(".context_length") and isinstance(value, int):
                info["context_length"] = value
                break
    _CAPS_CACHE[model] = info
    return info


def _infer_strengths(tag: str, caps: set[str]) -> dict[str, float] | None:
    """Task strengths from the model's name, extended by reported capabilities."""
    base = _base(tag)
    if "embed" in base or ("embedding" in caps and "completion" not in caps):
        return None
    if re.search(r"vl\b|llava|vision|moondream", base):
        return dict(_VISION)
    if re.search(r"coder|code", base):
        strengths = dict(_CODER)
    elif re.search(r"r1|qwq|think|reason", base):
        strengths = dict(_REASONER)
    else:
        strengths = dict(_GENERAL)
    # Capability tags add strengths; they never take any away.
    if "thinking" in caps:
        for task, fit in _REASONER.items():
            strengths[task] = max(strengths.get(task, 0.0), fit)
    if "vision" in caps:
        strengths["vision"] = max(strengths.get("vision", 0.0), _VISION["vision"])
    return strengths


def _size_profile(size_b: float | None) -> tuple[float, float]:
    """(quality, speed) from parameter count in billions."""
    if size_b is None or 4 < size_b <= 9:
        return 0.58, 0.80
    if size_b <= 4:
        return 0.45, 0.90
    if size_b <= 15:
        return 0.70, 0.50
    return 0.75, 0.25  # won't fit an 8 GB card; partly runs on the CPU


def discover(installed: list[str]) -> list[ModelSpec]:
    """Profiles for installed models the catalogue doesn't know about."""
    known = {
        resolve_installed(spec, installed)
        for spec in CATALOG
        if spec.provider == "ollama"
    }
    specs: list[ModelSpec] = []
    for tag in installed:
        if tag in known:
            continue
        info = capabilities(tag)
        strengths = _infer_strengths(tag, info["caps"])
        if strengths is None:
            continue
        quality, speed = _size_profile(info["parameter_size"] or _size_from_tag(tag))
        specs.append(
            ModelSpec(
                tag, "ollama",
                {task: round(fit * DISCOVERED_SCALE, 4) for task, fit in strengths.items()},
                quality=quality, speed=speed,
                note="profile inferred from name", discovered=True,
            )
        )
    return specs
```

In `plan()`, change `    for spec in CATALOG:` to:

```python
    for spec in CATALOG + discover(installed):
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=. python tests/test_providers.py -v` — Expected: 26 tests OK.
Run: `PYTHONPATH=. python tests/test_router.py -v` — Expected: 26 tests OK (catalogue-only inputs never call `capabilities`).

- [ ] **Step 5: Commit**

```bash
git add providers.py tests/test_providers.py
git commit -m "Let Auto pick installed models that aren't in the catalogue

providers.capabilities() reads Ollama's /api/show (tags, context window,
size). discover() infers a profile for every uncatalogued installed model
from its name and capabilities, scaled to 90% so hand-tuned entries win
ties. Embedding models are never offered."
```

---

### Task 5: Streamed `/api/chat` client with timeouts and metrics

**Files:**
- Modify: `engine.py` (imports; new block after `get_router()`)
- Create: `tests/fakes.py`
- Create: `tests/test_engine.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `engine.OLLAMA_CHAT_URL`, `engine.OLLAMA_TIMEOUT = (5, 120)`, `engine.NUM_CTX`, `engine.ChatReply(text: str, thinking: str | None, metrics: dict)`, `engine._ollama_chat(model, messages, *, temperature=0.7, num_ctx=NUM_CTX, think=False, on_token=None, on_thinking=None) -> ChatReply`, `engine._guard(callback) -> callback | None`, `engine._CallbackRaised` (has `.original`). Metrics keys: `ttft_ms, load_ms, prompt_tokens, eval_tokens, tokens_per_s`. Test helpers `fakes.FakeResponse`, `fakes.FakeOllama(scripts)` (`.calls`: list of `{url, json, timeout, stream}`), `fakes.stream(*tokens, thinking=(), done_extra=None) -> list[dict]`.

- [ ] **Step 1: Create the fakes**

Create `tests/fakes.py`:

```python
"""Stand-ins for Ollama, the router and the retriever, shared by the tests.

Not a test module itself (CI only runs tests/test_*.py).
"""

import json as _json

import requests


class FakeResponse:
    def __init__(self, lines, status=200):
        self._lines = [_json.dumps(l).encode("utf-8") if isinstance(l, dict) else l for l in lines]
        self.status_code = status
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_lines(self):
        yield from self._lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False


def stream(*tokens, thinking=(), done_extra=None):
    """NDJSON chunks as /api/chat streams them: thinking, then content, then done."""
    lines = [{"message": {"role": "assistant", "thinking": t}} for t in thinking]
    lines += [{"message": {"role": "assistant", "content": t}} for t in tokens]
    final = {
        "done": True,
        "message": {"role": "assistant", "content": ""},
        "load_duration": 2_000_000_000,      # 2 s -> a cold load
        "prompt_eval_count": 50,
        "eval_count": 20,
        "eval_duration": 1_000_000_000,      # 1 s -> 20 tok/s
    }
    final.update(done_extra or {})
    return lines + [final]


class FakeOllama:
    """Replaces requests.post. `scripts` maps model -> list of chunks, or an
    exception to raise for that model."""

    def __init__(self, scripts):
        self.scripts = scripts
        self.calls = []

    def __call__(self, url, json=None, timeout=None, stream=False, **_):
        self.calls.append({"url": url, "json": json, "timeout": timeout, "stream": stream})
        script = self.scripts[json["model"]]
        if isinstance(script, Exception):
            raise script
        return FakeResponse(script)
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_engine.py`:

```python
import os
import tempfile
import unittest
from unittest import mock

_TMP = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
os.environ["NEXUS_DB"] = os.path.join(_TMP.name, "engine-test.db")

import engine  # noqa: E402
from fakes import FakeOllama, stream  # noqa: E402

MESSAGES = [{"role": "user", "content": "hi"}]


class OllamaChatTests(unittest.TestCase):
    def use(self, scripts):
        self.ollama = FakeOllama(scripts)
        patcher = mock.patch.object(engine.requests, "post", self.ollama)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_streams_tokens_and_reports_metrics(self):
        self.use({"a": stream("Hel", "lo")})
        seen = []
        reply = engine._ollama_chat("a", MESSAGES, on_token=seen.append)
        self.assertEqual(reply.text, "Hello")
        self.assertEqual(seen, ["Hel", "lo"])
        self.assertIsNone(reply.thinking)
        self.assertEqual(reply.metrics["tokens_per_s"], 20.0)
        self.assertEqual(reply.metrics["load_ms"], 2000.0)
        self.assertEqual(reply.metrics["prompt_tokens"], 50)
        self.assertEqual(reply.metrics["eval_tokens"], 20)
        self.assertIsNotNone(reply.metrics["ttft_ms"])

    def test_payload_uses_chat_endpoint_window_and_timeouts(self):
        self.use({"a": stream("ok")})
        engine._ollama_chat("a", MESSAGES, temperature=0.2, num_ctx=4096)
        call = self.ollama.calls[0]
        self.assertTrue(call["url"].endswith("/api/chat"))
        self.assertTrue(call["stream"])
        self.assertEqual(call["timeout"], (5, 120))
        self.assertEqual(call["json"]["messages"], MESSAGES)
        self.assertEqual(call["json"]["options"], {"temperature": 0.2, "num_ctx": 4096})
        self.assertNotIn("think", call["json"])

    def test_think_flag_is_sent_only_when_asked(self):
        self.use({"a": stream("ok")})
        engine._ollama_chat("a", MESSAGES, think=True)
        self.assertIs(self.ollama.calls[0]["json"]["think"], True)

    def test_native_thinking_is_kept_separate(self):
        self.use({"a": stream("Answer", thinking=("step one ", "step two"))})
        thoughts = []
        reply = engine._ollama_chat("a", MESSAGES, on_thinking=thoughts.append)
        self.assertEqual(reply.text, "Answer")
        self.assertEqual(reply.thinking, "step one step two")
        self.assertEqual(thoughts, ["step one ", "step two"])

    def test_inline_think_block_is_moved_out_of_the_answer(self):
        self.use({"a": stream("<think>plan it</think>", "\n\nDone")})
        reply = engine._ollama_chat("a", MESSAGES)
        self.assertEqual(reply.text, "Done")
        self.assertEqual(reply.thinking, "plan it")

    def test_close_tag_only(self):
        self.use({"a": stream("plan it</think>Done")})
        reply = engine._ollama_chat("a", MESSAGES)
        self.assertEqual((reply.text, reply.thinking), ("Done", "plan it"))

    def test_error_chunk_raises(self):
        self.use({"a": [{"error": "model not found"}]})
        with self.assertRaisesRegex(RuntimeError, "model not found"):
            engine._ollama_chat("a", MESSAGES)

    def test_empty_reply_raises(self):
        self.use({"a": stream()})
        with self.assertRaisesRegex(RuntimeError, "empty"):
            engine._ollama_chat("a", MESSAGES)

    def test_guard_wraps_callback_errors(self):
        def boom(_):
            raise ValueError("ui gone")

        with self.assertRaises(engine._CallbackRaised) as caught:
            engine._guard(boom)("x")
        self.assertIsInstance(caught.exception.original, ValueError)
        self.assertIsNone(engine._guard(None))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run to verify it fails**

Run: `PYTHONPATH=. python tests/test_engine.py -v` — Expected: `AttributeError: module 'engine' has no attribute '_ollama_chat'`.

- [ ] **Step 4: Implement the client**

In `engine.py`, change the imports at the top to add `os` and `re`:

```python
import json
import os
import re
import time
```

Insert this block directly after the `get_router()` function:

```python
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
# 5 s to connect, then at most 120 s of silence between streamed chunks. A long
# answer is fine while tokens keep arriving; a stuck model fails and the chain
# moves on. Loading a 7B model (~30 s here) happens before the first chunk.
OLLAMA_TIMEOUT = (5, 120)
# Ollama's default window is 2k-4k tokens depending on version, and it silently
# drops the front of anything longer. Always ask for this much explicitly.
NUM_CTX = int(os.environ.get("NEXUS_NUM_CTX") or 8192)

_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.S)


@dataclass
class ChatReply:
    text: str
    thinking: str | None
    metrics: dict[str, Any]


class _CallbackRaised(Exception):
    """An exception from the caller's on_token/on_thinking callback, carried
    out of the fallback chain so it is never mistaken for the model failing."""

    def __init__(self, original: Exception):
        super().__init__(repr(original))
        self.original = original


def _guard(callback: Callable[[str], None] | None) -> Callable[[str], None] | None:
    if callback is None:
        return None

    def call(text: str) -> None:
        try:
            callback(text)
        except Exception as exc:
            raise _CallbackRaised(exc) from exc

    return call


def _split_inline_thinking(text: str, thinking: str | None) -> tuple[str, str | None]:
    """Older Ollama versions put reasoning inline as <think>…</think>."""
    parts = [thinking] if thinking else []
    match = _THINK_BLOCK.search(text)
    if match:
        parts.append(match.group(1).strip())
        text = _THINK_BLOCK.sub("", text, count=1)
    elif "</think>" in text:
        # Some chat templates open the block in the prompt, so only the close arrives.
        head, _, text = text.partition("</think>")
        parts.append(head.strip())
    joined = "\n".join(p for p in parts if p)
    return text.strip(), joined or None


def _metrics(final: dict[str, Any], sent: float, first: float | None) -> dict[str, Any]:
    eval_count = final.get("eval_count")
    eval_seconds = (final.get("eval_duration") or 0) / 1e9
    load_ns = final.get("load_duration")
    return {
        "ttft_ms": round((first - sent) * 1000, 1) if first is not None else None,
        "load_ms": round(load_ns / 1e6, 1) if load_ns is not None else None,
        "prompt_tokens": final.get("prompt_eval_count"),
        "eval_tokens": eval_count,
        "tokens_per_s": (
            round(eval_count / eval_seconds, 2) if eval_count and eval_seconds > 0 else None
        ),
    }


def _ollama_chat(
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.7,
    num_ctx: int = NUM_CTX,
    think: bool = False,
    on_token: Callable[[str], None] | None = None,
    on_thinking: Callable[[str], None] | None = None,
) -> ChatReply:
    """One streamed /api/chat call.

    Raises on HTTP errors, error chunks and empty replies. Callback exceptions
    propagate unchanged; the `with` block closes the socket on the way out,
    which is what makes Ollama stop generating.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": float(temperature), "num_ctx": int(num_ctx)},
    }
    if think:
        payload["think"] = True

    sent = time.monotonic()
    first: float | None = None
    parts: list[str] = []
    thoughts: list[str] = []
    final: dict[str, Any] = {}
    with requests.post(
        OLLAMA_CHAT_URL, json=payload, timeout=OLLAMA_TIMEOUT, stream=True
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            if chunk.get("error"):
                raise RuntimeError(f"{model}: {chunk['error']}")
            message = chunk.get("message") or {}
            thought = message.get("thinking") or ""
            token = message.get("content") or ""
            if (thought or token) and first is None:
                first = time.monotonic()
            if thought:
                thoughts.append(thought)
                if on_thinking:
                    on_thinking(thought)
            if token:
                parts.append(token)
                if on_token:
                    on_token(token)
            if chunk.get("done"):
                final = chunk
                break

    text, thinking = _split_inline_thinking("".join(parts), "".join(thoughts) or None)
    if not text:
        raise RuntimeError(f"{model} returned an empty response")
    return ChatReply(text, thinking, _metrics(final, sent, first))
```

- [ ] **Step 5: Run to verify it passes**

Run: `PYTHONPATH=. python tests/test_engine.py -v` — Expected: 9 tests OK.

- [ ] **Step 6: Commit**

```bash
git add engine.py tests/fakes.py tests/test_engine.py
git commit -m "Add a streamed /api/chat client with an explicit context window

Sets num_ctx instead of inheriting Ollama's 2-4k default, replaces the flat
300 s timeout with 5 s connect / 120 s between chunks, separates reasoning
from the answer (native field or inline <think>), and reports load time,
time to first token and tokens/s."
```

---

### Task 6: `answer()` — history, context budget, follow-ups, turn recording

**Files:**
- Modify: `engine.py` (imports; constants; new helpers; rewrite `_base_result` and `answer`; delete `OLLAMA_GENERATE_URL`, `_ollama_generate`, `_generate`)
- Modify: `tests/fakes.py` (add router/retriever fakes)
- Modify: `tests/test_engine.py` (add `AnswerTests`)

**Interfaces:**
- Consumes: `store.record_turn(row)`, `store.recent_turns(limit)`, `store.close()` (Task 3); `providers.capabilities(model)` (Task 4); `_ollama_chat`, `_guard`, `_CallbackRaised`, `NUM_CTX` (Task 5); `rag_pipeline.build_prompt(question, chunks)`, `rag_pipeline.get_retriever()`.
- Produces: `engine.answer(question, base_dir=None, options=None, on_token=None, on_thinking=None, history=None, chat_id=None) -> dict` with added keys `thinking, truncated, chunks_dropped, stopped, metrics` (`metrics` = `ttft_ms, total_ms, load_ms, prompt_tokens, eval_tokens, tokens_per_s`); `engine.estimate_tokens(text) -> int`; `engine.context_window(model) -> int`; `engine.fit_to_window(question, chunks, history, num_ctx, use_rag) -> Fitted(messages, kept_chunks, chunks_dropped, over_budget)`; constants `REPLY_RESERVE = 1024`, `TRUNCATION_RATIO = 0.98`, `FOLLOW_UP_WORDS = 12`. Fakes: `fakes.FakeRouter(task="general", chain=("a",), needs_rag=False)`, `fakes.FakeRetriever(chunks)` (`.queries`), `fakes.chunk(i, words=50) -> dict`.

- [ ] **Step 1: Extend the fakes**

Append to `tests/fakes.py`:

```python
class FakeRouter:
    def __init__(self, task="general", chain=("a",), needs_rag=False):
        self.task, self.chain, self.needs_rag = task, list(chain), needs_rag

    def route(self, query, available_models=None, force_task=None):
        task = force_task or self.task
        chain = [
            {"model": m, "provider": "ollama", "score": round(1.0 - i / 10, 2),
             "reason": f"{m}: test"}
            for i, m in enumerate(self.chain)
        ]
        return {
            "task": task, "model": self.chain[0] if self.chain else None,
            "provider": "ollama", "chain": chain, "complexity": 0.2,
            "available": {"ollama": list(self.chain)}, "needs_rag": self.needs_rag,
            "confidence": 1.0, "scores": {task: 1.0}, "auto_task": self.task,
            "forced": bool(force_task), "reason": "test",
        }


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.queries = []

    def query(self, question, top_k=10, rerank=True, **_):
        self.queries.append(question)
        return self.chunks[:top_k]


def chunk(i, words=50):
    return {
        "text": f"chunk{i} " + "word " * words,
        "meta": {"source": f"doc{i}.md", "chunk_index": 0},
        "score": 1.0 / (i + 1), "vector_rank": i, "bm25_rank": i,
    }
```

- [ ] **Step 2: Write the failing tests**

In `tests/test_engine.py` change the fakes import to
`from fakes import FakeOllama, FakeRetriever, FakeRouter, chunk, stream  # noqa: E402`, add `import providers  # noqa: E402`, `import store  # noqa: E402` and `from pathlib import Path`, then append before the `__main__` block:

```python
NO_CAPS = {"caps": set(), "context_length": None, "parameter_size": None}


def tearDownModule():
    store.close()
    _TMP.cleanup()


class AnswerTests(unittest.TestCase):
    def use(self, *, chain=("a",), task="general", needs_rag=False, scripts=None,
            caps=None, chunks=()):
        self.router = FakeRouter(task=task, chain=chain, needs_rag=needs_rag)
        self.retriever = FakeRetriever(chunks)
        self.ollama = FakeOllama(scripts or {m: stream("ok") for m in chain})
        caps = caps or {}
        for patcher in (
            mock.patch.object(engine, "get_router", lambda: self.router),
            mock.patch.object(engine, "get_retriever", lambda: self.retriever),
            mock.patch.object(engine.requests, "post", self.ollama),
            mock.patch.object(providers, "capabilities",
                              lambda m: {**NO_CAPS, **caps.get(m, {})}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def sent(self, call=-1):
        return self.ollama.calls[call]["json"]

    def test_falls_through_to_the_next_model(self):
        self.use(chain=("a", "b"), scripts={"a": ConnectionError("down"), "b": stream("ok")})
        result = engine.answer("hello")
        self.assertEqual(result["model"], "b")
        self.assertEqual([a["model"] for a in result["attempts"]], ["a", "b"])
        self.assertIn("Auto-switched", result["info"])

    def test_pinned_model_goes_first(self):
        self.use(chain=("a", "b"))
        engine.answer("hello", options=engine.Options(force_model="b"))
        self.assertEqual(self.sent(0)["model"], "b")

    def test_rag_never_skips_retrieval(self):
        self.use(needs_rag=True, chunks=[chunk(0)])
        result = engine.answer("what does the readme say",
                               options=engine.Options(rag_mode="never"))
        self.assertEqual(self.retriever.queries, [])
        self.assertEqual(result["sources"], [])

    def test_rag_always_grounds_the_newest_message(self):
        self.use(needs_rag=False, chunks=[chunk(0)])
        result = engine.answer("hello", options=engine.Options(rag_mode="always"))
        self.assertEqual(self.retriever.queries, ["hello"])
        self.assertEqual(len(result["sources"]), 1)
        self.assertIn("chunk0", self.sent()["messages"][-1]["content"])

    def test_history_goes_before_the_question_in_order(self):
        self.use()
        history = [
            {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"},
        ]
        engine.answer("q3", history=history)
        self.assertEqual(self.sent()["messages"],
                         history + [{"role": "user", "content": "q3"}])

    def test_history_is_trimmed_newest_first_to_fit(self):
        # context 1400 -> budget 376 tokens; each message costs 100 + 4.
        self.use(caps={"a": {"context_length": 1400}})
        history = [{"role": "user" if i % 2 == 0 else "assistant",
                    "content": f"m{i} " + "x" * 297} for i in range(10)]
        engine.answer("q?", history=history)
        self.assertEqual(self.sent()["messages"],
                         history[-3:] + [{"role": "user", "content": "q?"}])
        self.assertEqual(self.sent()["options"]["num_ctx"], 1400)

    def test_chunks_that_do_not_fit_are_dropped_and_counted(self):
        self.use(needs_rag=True, caps={"a": {"context_length": 2048}},
                 chunks=[chunk(i) for i in range(20)])
        result = engine.answer("what does the readme say",
                               options=engine.Options(top_k=20))
        self.assertGreater(result["chunks_dropped"], 0)
        self.assertEqual(len(result["sources"]) + result["chunks_dropped"], 20)
        self.assertLessEqual(
            engine.estimate_tokens(self.sent()["messages"][-1]["content"]),
            2048 - engine.REPLY_RESERVE,
        )
        self.assertIn("left out", result["info"])

    def test_truncation_is_flagged_when_the_window_filled(self):
        self.use(caps={"a": {"context_length": 2048}},
                 scripts={"a": stream("ok", done_extra={"prompt_eval_count": 2030})})
        result = engine.answer("hello")
        self.assertTrue(result["truncated"])
        self.assertIn("context window", result["info"])

    def test_no_truncation_flag_normally(self):
        self.use()
        self.assertFalse(engine.answer("hello")["truncated"])

    def test_think_is_sent_only_to_thinking_models(self):
        self.use(chain=("a", "b"),
                 scripts={"a": ConnectionError("down"), "b": stream("ok")},
                 caps={"a": {"caps": {"completion", "thinking"}}})
        engine.answer("hello")
        self.assertIs(self.sent(0).get("think"), True)
        self.assertNotIn("think", self.sent(1))

    def test_thinking_reaches_the_result(self):
        self.use(scripts={"a": stream("ok", thinking=("hmm",))})
        self.assertEqual(engine.answer("hello")["thinking"], "hmm")

    def test_callback_errors_are_not_model_failures(self):
        self.use(chain=("a", "b"))

        class UiGone(Exception):
            pass

        def on_token(_):
            raise UiGone()

        with self.assertRaises(UiGone):
            engine.answer("hello", on_token=on_token)
        self.assertEqual(len(self.ollama.calls), 1)  # b was never tried

    def test_follow_up_borrows_the_previous_question_for_search_only(self):
        self.use(chunks=[chunk(0)])
        history = [{"role": "user", "content": "Which embedding model does NEXUS use?"},
                   {"role": "assistant", "content": "all-MiniLM-L6-v2."}]
        engine.answer("and its dimension?", history=history,
                      options=engine.Options(rag_mode="always"))
        self.assertEqual(self.retriever.queries,
                         ["Which embedding model does NEXUS use? and its dimension?"])
        self.assertIn("User question: and its dimension?",
                      self.sent()["messages"][-1]["content"])

    def test_one_turn_is_recorded_per_answer(self):
        self.use()
        engine.answer("hello", chat_id="chat-1")
        turn = store.recent_turns(1)[0]
        self.assertEqual((turn["model"], turn["chat_id"]), ("a", "chat-1"))
        self.assertEqual(turn["tokens_per_s"], 20.0)
        self.assertIsNotNone(turn["total_ms"])

    def test_a_turn_is_recorded_even_when_every_model_fails(self):
        self.use(scripts={"a": ConnectionError("down")})
        with self.assertRaises(RuntimeError):
            engine.answer("hello")
        turn = store.recent_turns(1)[0]
        self.assertIsNone(turn["model"])
        self.assertTrue(turn["error"].startswith("Every available model failed"))

    def test_system_agent_only_previews(self):
        self.use(task="system_agent")
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "empty").mkdir()
            result = engine.answer(f'remove empty folders in "{d}"')
            self.assertTrue(result["requires_confirmation"])
            self.assertTrue((Path(d) / "empty").is_dir())
        self.assertEqual(store.recent_turns(1)[0]["model"], "pc-toolkit")
```

- [ ] **Step 3: Run to verify it fails**

Run: `PYTHONPATH=. python tests/test_engine.py -v` — Expected: `AnswerTests` fail/error (`get_retriever` not in engine; unexpected `history` kwarg).

- [ ] **Step 4: Implement**

In `engine.py`:

1. Imports — make the header imports:

```python
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import requests

import pc_agent
import providers
import store
from rag_pipeline import DEFAULT_TOP_K, build_prompt, get_retriever
from router import TaskRouter
```

2. Delete `OLLAMA_GENERATE_URL = ...`, the whole `_ollama_generate` function and the whole `_generate` function.

3. Append after `_ollama_chat` (end of the Task 5 block):

```python
REPLY_RESERVE = 1024     # tokens kept free for the model's answer
TRUNCATION_RATIO = 0.98  # prompt_eval_count this close to num_ctx => probably cut off
FOLLOW_UP_WORDS = 12     # a message shorter than this, with history, is a follow-up


def estimate_tokens(text: str) -> int:
    """Deliberately pessimistic: ~3 chars per token, where English on Llama/Qwen
    tokenizers runs ~4. Over-estimating means sending slightly less — never
    having Ollama silently cut the front off."""
    return math.ceil(len(text) / 3)


def context_window(model: str) -> int:
    reported = providers.capabilities(model).get("context_length")
    return min(NUM_CTX, reported) if reported else NUM_CTX


@dataclass
class Fitted:
    messages: list[dict[str, str]]
    kept_chunks: list[dict[str, Any]]
    chunks_dropped: int
    over_budget: bool


def fit_to_window(
    question: str,
    chunks: list[dict[str, Any]],
    history: list[dict[str, str]],
    num_ctx: int,
    use_rag: bool,
) -> Fitted:
    """Build one model's chat messages so they fit its context window.

    Priority: the question (with the RAG template) always; then retrieved
    chunks in rank order; then history, newest first, whole messages only.
    """
    budget = num_ctx - REPLY_RESERVE

    def render(kept: list[dict[str, Any]]) -> str:
        return build_prompt(question, kept) if use_rag else question

    used = estimate_tokens(render([]))
    over = used > budget  # sent anyway: refusing would be worse than a cut prompt

    kept: list[dict[str, Any]] = []
    if use_rag and not over:
        for candidate in chunks:
            cost = estimate_tokens(render(kept + [candidate]))
            if cost > budget:
                break
            kept.append(candidate)
            used = cost

    past: list[dict[str, str]] = []
    if not over:
        for msg in reversed(history):
            cost = estimate_tokens(msg["content"]) + 4  # role/framing overhead
            if used + cost > budget:
                break
            past.insert(0, {"role": msg["role"], "content": msg["content"]})
            used += cost

    return Fitted(
        past + [{"role": "user", "content": render(kept)}],
        kept,
        len(chunks) - len(kept),
        over,
    )


def _retrieval_query(question: str, history: list[dict[str, str]]) -> str:
    """"and the second one?" retrieves nothing useful alone — borrow the
    previous question. Only the search query changes, never the prompt."""
    if history and len(question.split()) < FOLLOW_UP_WORDS:
        previous = next((m["content"] for m in reversed(history) if m["role"] == "user"), None)
        if previous:
            return f"{previous} {question}"
    return question


def _source(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": chunk["meta"].get("source", "unknown"),
        "chunk_index": chunk["meta"].get("chunk_index"),
        "score": chunk.get("score"),
        "rerank_score": chunk.get("rerank_score"),
        "vector_rank": chunk.get("vector_rank"),
        "bm25_rank": chunk.get("bm25_rank"),
        "text": chunk["text"],
    }


def _note(result: dict[str, Any], text: str) -> None:
    result["info"] = f"{result['info']} {text}" if result.get("info") else text


def _finish(
    result: dict[str, Any], started: float, chat_id: str | None, error: str | None = None
) -> None:
    """Stamp timings and record exactly one `turns` row. Never raises."""
    result["elapsed"] = time.monotonic() - started
    metrics = result["metrics"]
    metrics["total_ms"] = round(result["elapsed"] * 1000, 1)
    row = {
        "chat_id": chat_id,
        "task": result.get("task"),
        "model": None if error else result.get("model"),
        "attempts": result.get("attempts"),
        "rag_used": bool(result.get("sources")),
        "chunks_dropped": result.get("chunks_dropped", 0),
        "truncated": result.get("truncated", False),
        "stopped": False,
        "error": error,
        **{k: metrics.get(k) for k in
           ("ttft_ms", "total_ms", "load_ms", "prompt_tokens", "eval_tokens", "tokens_per_s")},
    }
    try:
        store.record_turn(row)
    except Exception as exc:
        print(f"[nexus] could not record metrics: {exc}", file=sys.stderr)
```

4. In `_base_result`, add these entries to the returned dict (after `"prompt_chars": 0,`):

```python
        "thinking": None,
        "truncated": False,
        "chunks_dropped": 0,
        "stopped": False,
        "metrics": {
            "ttft_ms": None, "total_ms": None, "load_ms": None,
            "prompt_tokens": None, "eval_tokens": None, "tokens_per_s": None,
        },
```

5. Replace the whole `answer()` function with:

```python
def answer(
    question: str,
    base_dir: Path | None = None,
    options: Options | None = None,
    on_token: Callable[[str], None] | None = None,
    on_thinking: Callable[[str], None] | None = None,
    history: list[dict[str, str]] | None = None,
    chat_id: str | None = None,
) -> dict[str, Any]:
    """Route `question`, then answer it with the best reachable model.

    `history` is the conversation so far ({"role", "content"}, oldest first);
    as much as fits the model's context window is sent, newest first. Returns
    the routing decision plus: answer, attempts, info, sources, thinking,
    truncated, chunks_dropped, metrics, requires_confirmation, pending, elapsed.
    One `turns` row is recorded per call. Exceptions raised by `on_token` /
    `on_thinking` propagate unchanged and are never treated as a model failure.
    """
    started = time.monotonic()
    base_dir = base_dir or PROJECT_DIR
    opts = (options or Options()).normalised()
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in (history or [])
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]

    decision = get_router().route(question, force_task=opts.force_task)
    result = _base_result(decision)

    if decision["task"] == "system_agent":
        outcome = pc_agent.handle(question, base_dir)
        result["answer"] = outcome["answer"]
        result["requires_confirmation"] = outcome["requires_confirmation"]
        result["pending"] = outcome["pending"]
        result["model"] = "pc-toolkit"
        result["provider"] = "toolkit"
        # The word "folder" trips the RAG signal, but a file operation never
        # retrieves anything -- don't claim it did.
        result["needs_rag"] = False
        _finish(result, started, chat_id)
        return result

    chain = decision.get("chain") or []
    if opts.force_model and chain:
        chain = _pin_model(chain, opts.force_model)
        result["chain"] = chain
        result["model"] = chain[0]["model"]
        result["provider"] = chain[0]["provider"]

    if not chain:
        result["answer"] = _no_model_message(decision)
        result["info"] = "No model was reachable for this request."
        _finish(result, started, chat_id, error=result["info"])
        return result

    # Retrieval gate: the router's keyword guess by default, or the override.
    if opts.rag_mode == "always":
        result["needs_rag"] = True
    elif opts.rag_mode == "never":
        result["needs_rag"] = False

    chunks: list[dict[str, Any]] = []
    if result["needs_rag"]:
        try:
            chunks = get_retriever().query(
                _retrieval_query(question, history), top_k=opts.top_k, rerank=opts.rerank
            )
            if not chunks:
                _note(result, "No local documents matched; answering without them.")
        except Exception as exc:
            _note(result, f"Local document search unavailable ({exc}); answering without it.")
            result["needs_rag"] = False

    token_cb, thinking_cb = _guard(on_token), _guard(on_thinking)
    errors: list[str] = []
    for step, candidate in enumerate(chain):
        model, provider = candidate["model"], candidate["provider"]
        if provider != "ollama":
            error = f"no generator for provider {provider!r}"
            result["attempts"].append({"model": model, "provider": provider, "error": error})
            errors.append(f"{model}: {error}")
            continue

        num_ctx = context_window(model)
        fitted = fit_to_window(question, chunks, history, num_ctx, result["needs_rag"])
        think = "thinking" in providers.capabilities(model).get("caps", set())
        try:
            reply = _ollama_chat(
                model, fitted.messages, temperature=opts.temperature, num_ctx=num_ctx,
                think=think, on_token=token_cb, on_thinking=thinking_cb,
            )
        except _CallbackRaised as wrapped:
            raise wrapped.original
        except Exception as exc:
            result["attempts"].append({"model": model, "provider": provider, "error": str(exc)})
            errors.append(f"{model}: {exc}")
            continue

        result["attempts"].append({"model": model, "provider": provider, "error": None})
        result.update(
            model=model,
            provider=provider,
            why=candidate.get("reason"),
            answer=reply.text,
            thinking=reply.thinking,
            sources=[_source(c) for c in fitted.kept_chunks],
            chunks_dropped=fitted.chunks_dropped,
            prompt_chars=sum(len(m["content"]) for m in fitted.messages),
        )
        result["metrics"].update(reply.metrics)
        if fitted.chunks_dropped:
            _note(result, f"{fitted.chunks_dropped} of {len(chunks)} retrieved chunks were "
                          f"left out to fit {model}'s {num_ctx}-token context window.")
        prompt_tokens = reply.metrics.get("prompt_tokens") or 0
        if fitted.over_budget or prompt_tokens >= TRUNCATION_RATIO * num_ctx:
            result["truncated"] = True
            _note(result, f"The prompt filled {model}'s {num_ctx}-token context window, "
                          "so the start of it may have been cut off.")
        if step > 0:
            _note(result, f"Auto-switched from {chain[0]['model']} to {model} — "
                          "first choice was unavailable.")
        _finish(result, started, chat_id)
        return result

    message = "Every available model failed for this request:\n  " + "\n  ".join(errors)
    _finish(result, started, chat_id, error=message)
    raise RuntimeError(message)
```

6. Update the module docstring's last paragraph to mention `history`:

```
Both the Streamlit UI (app.py) and the plain HTTP server (server.py) call
`answer()` so the routing/fallback behaviour stays in one place. Streamlit also
passes the conversation so far as `history`.
```

- [ ] **Step 5: Run to verify it passes**

Run: `PYTHONPATH=. python tests/test_engine.py -v` — Expected: 25 tests OK.
Run the whole suite: `for t in tests/test_*.py; do PYTHONPATH=. python "$t" || echo "FAIL $t"; done` — Expected: no `FAIL` lines.

- [ ] **Step 6: Commit**

```bash
git add engine.py tests/fakes.py tests/test_engine.py
git commit -m "Send conversation history and budget every prompt to its window

answer() now takes history and fits question, retrieved chunks and past
turns into each model's context window, in that priority. Dropped chunks
and a filled window are reported instead of being silently cut by Ollama.
Short follow-ups borrow the previous question for retrieval. Every call
records one turns row, including all-failed ones."
```

---

### Task 7: Streamlit — saved chats, history, Reasoning panel, Diagnostics

**Files:**
- Modify: `app.py`
- Create: `tests/test_app.py`

**Interfaces:**
- Consumes: `store.create_chat/list_chats/load_messages/append_message/delete_chat/recent_turns/model_stats` (Task 3); `providers.discover`, `providers.capabilities` (Task 4); `engine.answer(..., on_thinking=, history=, chat_id=)` (Task 6).
- Produces: sidebar buttons with keys `open_<chat_id>` and `del_<chat_id>`; session keys `chat_id`, `confirm_delete`; helpers `_persist(role, content, meta=None)`, `_history_for(regenerate: bool) -> list[dict]`, `render_reasoning(meta)`; `run_turn(question, opts, regenerate=False)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_app.py`:

```python
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_TMP = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
os.environ["NEXUS_DB"] = os.path.join(_TMP.name, "app-test.db")

from streamlit.testing.v1 import AppTest  # noqa: E402

import engine  # noqa: E402
import providers  # noqa: E402
import retrieve  # noqa: E402
import store  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")
CHAIN = [
    {"model": "a", "provider": "ollama", "score": 1.0, "reason": "a: test"},
    {"model": "b", "provider": "ollama", "score": 0.9, "reason": "b: test"},
]


class FakeRetriever:
    _doc_by_id: dict = {}
    _indexed_count = 0

    def __init__(self, *args, **kwargs):
        pass

    def _ensure_fresh(self):
        pass

    def refresh(self):
        pass

    def query(self, *args, **kwargs):
        return []


def tearDownModule():
    store.close()
    _TMP.cleanup()


class AppFlowTests(unittest.TestCase):
    def setUp(self):
        conn = store.connect()
        with conn:
            for table in ("messages", "chats", "turns"):
                conn.execute(f"DELETE FROM {table}")
        conn.close()
        self.calls = []
        for patcher in (
            mock.patch.object(engine, "answer", self.fake_answer),
            mock.patch.object(providers, "availability",
                              lambda *a, **k: {"ollama": ["a", "b"], "ollama_up": True,
                                               "loaded": []}),
            mock.patch.object(providers, "capabilities",
                              lambda m: {"caps": set(), "context_length": None,
                                         "parameter_size": None}),
            mock.patch.object(retrieve, "Retriever", FakeRetriever),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def fake_answer(self, question, base_dir=None, options=None, on_token=None,
                    on_thinking=None, history=None, chat_id=None):
        self.calls.append({"question": question, "options": options,
                           "history": history, "chat_id": chat_id})
        model = options.force_model if options and options.force_model else "a"
        if on_token:
            on_token(f"answer from {model}")
        return {
            "answer": f"answer from {model}", "model": model, "task": "general",
            "needs_rag": False, "sources": [], "elapsed": 0.1, "chain": CHAIN,
            "complexity": 0.1, "attempts": [{"model": model, "provider": "ollama", "error": None}],
            "prompt_chars": len(question), "info": None, "auto_task": "general",
            "thinking": None, "truncated": False, "chunks_dropped": 0, "metrics": {},
            "requires_confirmation": False, "pending": None,
        }

    def start(self):
        at = AppTest.from_file(APP, default_timeout=90)
        at.run()
        self.assertFalse(at.exception, at.exception)
        return at

    def ask(self, at, text):
        at.chat_input[0].set_value(text).run()
        self.assertFalse(at.exception, at.exception)

    def test_answer_is_saved_and_reopens_in_a_new_session(self):
        at = self.start()
        self.ask(at, "what is hybrid retrieval?")
        [chat] = store.list_chats()
        self.assertEqual(chat["title"], "what is hybrid retrieval?")
        self.assertEqual([m["role"] for m in store.load_messages(chat["id"])],
                         ["user", "assistant"])

        fresh = self.start()
        self.assertEqual(len(fresh.chat_message), 0)
        fresh.button(key=f"open_{chat['id']}").click().run()
        self.assertEqual(len(fresh.chat_message), 2)

    def test_follow_up_sends_the_earlier_turns(self):
        at = self.start()
        self.ask(at, "first question")
        self.ask(at, "and the second one?")
        last = self.calls[-1]
        self.assertEqual(last["history"], [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "answer from a"},
        ])
        self.assertEqual(last["chat_id"], store.list_chats()[0]["id"])

    def test_regenerate_hides_the_answer_being_redone(self):
        at = self.start()
        self.ask(at, "only question")
        at.button(key="regen_b").click().run()
        last = self.calls[-1]
        self.assertEqual(last["options"].force_model, "b")
        self.assertEqual(last["history"], [])

    def test_deleting_a_chat(self):
        at = self.start()
        self.ask(at, "to be deleted")
        [chat] = store.list_chats()
        at.button(key=f"del_{chat['id']}").click().run()
        at.button(key="confirm_delete_chat").click().run()
        self.assertEqual(store.list_chats(), [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=. python tests/test_app.py -v` — Expected: failures (no chats saved; `open_…` key not found; `history` is `None`).

- [ ] **Step 3: Imports and session state**

In `app.py`, change `import json` / `import time` block to:

```python
import io
import json
import sys
import time
```

and after `import providers` add `import store`.

After `_state("regenerate_with", None)` add:

```python
_state("chat_id", None)
_state("confirm_delete", None)


def _persist(role: str, content: str, meta: dict | None = None) -> None:
    """Add a message to the transcript and save it. Saving never blocks chatting."""
    message = {"role": role, "content": content}
    if meta is not None:
        message["meta"] = meta
    st.session_state.messages.append(message)
    try:
        if st.session_state.chat_id is None:
            st.session_state.chat_id = store.create_chat(content)
        store.append_message(st.session_state.chat_id, role, content, meta)
    except Exception as exc:
        print(f"[nexus] could not save message: {exc}", file=sys.stderr)


def _history_for(regenerate: bool) -> list[dict]:
    """The turns before the question being answered, without UI metadata.

    For a regenerate, the answer being redone and its original question are
    left out too, so the model never sees the answer it's asked to replace.
    """
    msgs = st.session_state.messages
    prior = msgs[:-1] if msgs and msgs[-1]["role"] == "user" else list(msgs)
    if (regenerate and len(prior) >= 2
            and prior[-1]["role"] == "assistant" and prior[-2]["role"] == "user"):
        prior = prior[:-2]
    return [{"role": m["role"], "content": m["content"]} for m in prior]
```

- [ ] **Step 4: Sidebar — New chat resets the chat id; Chats list**

In the `New chat` handler add `st.session_state.chat_id = None` after `st.session_state.last_question = None`.

Insert directly before `options = Options(`:

```python
with st.sidebar:
    st.markdown("#### Chats")
    try:
        recent_chats = store.list_chats(limit=20)
    except Exception as exc:
        recent_chats = []
        st.caption(f"Saved chats unavailable ({exc})")
    if not recent_chats:
        st.caption("Chats you start are saved here.")
    for chat in recent_chats:
        c_open, c_del = st.columns([5, 1])
        mark = "▸ " if chat["id"] == st.session_state.chat_id else ""
        if c_open.button(f"{mark}{chat['title']}", key=f"open_{chat['id']}",
                         help=_ago(chat["updated"]), width='stretch'):
            st.session_state.messages = store.load_messages(chat["id"])
            st.session_state.chat_id = chat["id"]
            st.session_state.pending_action = None
            st.session_state.last_question = None
            st.rerun()
        if c_del.button("✕", key=f"del_{chat['id']}", help="Delete this chat"):
            st.session_state.confirm_delete = chat["id"]
            st.rerun()

    doomed = next((c for c in recent_chats if c["id"] == st.session_state.confirm_delete), None)
    if doomed:
        st.warning(f"Delete “{doomed['title']}”? This can't be undone.")
        d_yes, d_no = st.columns(2)
        if d_yes.button("Delete", key="confirm_delete_chat", type="primary"):
            store.delete_chat(doomed["id"])
            if st.session_state.chat_id == doomed["id"]:
                st.session_state.messages = []
                st.session_state.chat_id = None
                st.session_state.pending_action = None
            st.session_state.confirm_delete = None
            st.rerun()
        if d_no.button("Keep", key="cancel_delete_chat"):
            st.session_state.confirm_delete = None
            st.rerun()
```

Because the sidebar code runs before the helper section, define `_ago` next to `_persist` (after `_history_for`):

```python
def _ago(ts: float) -> str:
    minutes = int((time.time() - ts) // 60)
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes} min ago"
    if minutes < 1440:
        return f"{minutes // 60} h ago"
    return f"{minutes // 1440} d ago"
```

- [ ] **Step 5: Reasoning panel and `run_turn`**

Add after `render_sources`:

```python
def render_reasoning(meta: dict) -> None:
    thinking = meta.get("thinking")
    if thinking:
        with st.expander("Reasoning", expanded=False):
            st.markdown(thinking)


def _secs(ms) -> float | None:
    return None if ms is None else round(ms / 1000, 2)


def _turn_row(turn: dict) -> dict:
    failed = sum(1 for a in turn.get("attempts") or [] if a.get("error"))
    flags = [name for name, on in (("truncated", turn.get("truncated")),
                                   ("stopped", turn.get("stopped")),
                                   ("error", turn.get("error"))) if on]
    return {
        "when": datetime.fromtimestamp(turn["ts"]).strftime("%H:%M:%S"),
        "task": turn.get("task"),
        "model": turn.get("model") or "—",
        "total s": _secs(turn.get("total_ms")),
        "first token s": _secs(turn.get("ttft_ms")),
        "tok/s": turn.get("tokens_per_s"),
        "cold load": bool((turn.get("load_ms") or 0) > 1000),
        "fallbacks": failed,
        "flags": ", ".join(flags),
    }
```

Replace the whole `run_turn` function (from `def run_turn(question: str, opts: Options) -> None:` through `        st.session_state.pending_action = result["pending"]`) with:

```python
def run_turn(question: str, opts: Options, regenerate: bool = False) -> None:
    """Generate one answer, streaming it (and any reasoning) into the transcript."""
    history = _history_for(regenerate)
    with st.chat_message("assistant"):
        think_slot = st.empty()
        placeholder = st.empty()
        buffer: list[str] = []
        thoughts: list[str] = []

        def on_thinking(text: str) -> None:
            thoughts.append(text)
            with think_slot.container():
                with st.expander("Reasoning…", expanded=True):
                    st.markdown("".join(thoughts))
            if not buffer:
                placeholder.markdown("_thinking…_")

        def on_token(tok: str) -> None:
            buffer.append(tok)
            placeholder.markdown("".join(buffer) + "▌")

        try:
            result = answer(
                question, options=opts, on_token=on_token, on_thinking=on_thinking,
                history=history, chat_id=st.session_state.chat_id,
            )
        except Exception as exc:
            think_slot.empty()
            placeholder.empty()
            st.error(f"NEXUS could not answer: {exc}")
            return

        meta = {
            k: result.get(k)
            for k in (
                "model", "task", "needs_rag", "sources", "elapsed", "chain",
                "complexity", "attempts", "prompt_chars", "info", "auto_task",
                "thinking", "truncated", "chunks_dropped", "metrics",
            )
        }
        meta["forced_model"] = bool(opts.force_model)
        with think_slot.container():
            render_reasoning(meta)
        placeholder.markdown(result.get("answer", ""))
        render_badges(meta)
        if result.get("info"):
            st.info(result["info"])
        render_why(meta)
        render_sources(meta)

    _persist("assistant", result.get("answer", ""), meta)
    if result.get("requires_confirmation"):
        st.session_state.pending_action = result["pending"]
```

- [ ] **Step 6: Route every transcript write through `_persist`**

Make these exact replacements in the Chat tab:

| old | new |
|---|---|
| `                st.session_state.messages.append({"role": "user", "content": text})` (starter buttons) | `                _persist("user", text)` |
| `        run_turn(queued, turn_options)` | `        run_turn(queued, turn_options, regenerate=bool(alt))` |
| `                st.session_state.messages.append({"role": "assistant", "content": msg})` (Apply) | `                _persist("assistant", msg)` |
| the 3-line `st.session_state.messages.append(` … `"Cancelled — nothing on disk changed."}` … `)` (Cancel) | `                _persist("assistant", "Cancelled — nothing on disk changed.")` |
| the 3-line `st.session_state.messages.append(` … `{"role": "user", "content": question}` … `)` (regenerate) | `                        _persist("user", question)` |
| `        st.session_state.messages.append({"role": "user", "content": prompt})` (chat input) | `        _persist("user", prompt)` |

Replace the transcript loop:

```python
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            meta = message.get("meta")
            if meta:
                render_reasoning(meta)
            st.markdown(message["content"])
            if meta:
                render_badges(meta)
                render_why(meta)
                render_sources(meta)
```

Confirm nothing else appends directly: `grep -n "messages.append" app.py` — Expected: only the line inside `_persist`.

- [ ] **Step 7: Diagnostics additions**

Append at the end of `app.py` (inside `with tab_diag:`, after the routing-decisions block, 4-space indent):

```python
    st.divider()
    st.markdown("##### Recent answers")
    try:
        turns = store.recent_turns(20)
    except Exception as exc:
        turns = []
        st.caption(f"metrics unavailable ({exc})")
    if turns:
        st.dataframe([_turn_row(t) for t in turns], width='stretch', hide_index=True)
    else:
        st.caption("No answers recorded yet.")

    st.markdown("##### Per model")
    st.caption("Over the last 500 answers. Failure rate counts every attempt, "
               "including ones the next model in the chain recovered from.")
    try:
        stats = store.model_stats()
    except Exception:
        stats = []
    if stats:
        st.dataframe(
            [
                {
                    "model": s["model"],
                    "answers": s["answers"],
                    "median s": _secs(s["median_ms"]),
                    "median tok/s": s["median_tokens_per_s"],
                    "failure rate": f"{s['failure_rate']:.0%}",
                    "cold loads": s["cold_loads"],
                }
                for s in stats
            ],
            width='stretch',
            hide_index=True,
        )
    else:
        st.caption("Nothing yet.")

    st.markdown("##### Discovered models")
    discovered = providers.discover(installed)
    if discovered:
        st.caption("Installed but not in the catalogue — profiled from name, size and "
                   "reported capabilities, then scaled to 90%.")
        st.dataframe(
            [
                {
                    "model": spec.name,
                    "profile": ", ".join(
                        f"{task} {fit:.2f}"
                        for task, fit in sorted(spec.strengths.items(), key=lambda kv: -kv[1])
                    ),
                    "capabilities": ", ".join(
                        sorted(providers.capabilities(spec.name)["caps"])
                    ) or "unknown",
                }
                for spec in discovered
            ],
            width='stretch',
            hide_index=True,
        )
    else:
        st.caption("Every installed model is in the catalogue.")
```

- [ ] **Step 8: Run to verify it passes**

Run: `PYTHONPATH=. python tests/test_app.py -v` — Expected: 4 tests OK. (If `AppTest` reports a timeout on the first run, it is the one-off import of sentence-transformers; `default_timeout=90` covers it.)

- [ ] **Step 9: Manual check**

Launch the app with the `run` skill (or `python -m streamlit run app.py --server.headless true --server.port 8501` in the background) and open it in the built-in browser. Ask two questions, confirm the chat appears in the sidebar, click **New chat**, reopen the first chat, and check the Diagnostics tab shows the new sections. Stop the server.

- [ ] **Step 10: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "Save chats, send history, and show reasoning and metrics in the UI

The sidebar lists saved chats to reopen or delete. Each question is sent
with the earlier turns; a regenerate hides the answer being redone.
Reasoning models stream into a collapsed Reasoning panel, and Diagnostics
shows recent answers, per-model medians and failure rates, and any
auto-profiled models."
```

---

### Task 8: Stop button (spike first)

**Files:**
- Throwaway (scratchpad, not committed): `slow_ollama.py`, `spike_app.py`, `spike.log`
- Modify: `app.py` (`render_badges`, `render_why`, `run_turn`; new `_save_stopped`)

**Interfaces:**
- Consumes: `engine._ollama_chat`, `engine._guard`, `engine.OLLAMA_CHAT_URL` (Task 5); `_persist` (Task 7); `store.record_turn`.
- Produces: button key `stop_generation`; stopped messages carry `meta["stopped"] = True`; a `turns` row with `stopped = 1`.

- [ ] **Step 1: Write the spike server** (scratchpad `slow_ollama.py`)

```python
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG = sys.argv[1]


def log(msg):
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        i = 0
        try:
            for i in range(100):
                line = (json.dumps({"message": {"content": f"tok{i} "}}) + "\n").encode()
                self.wfile.write(f"{len(line):x}\r\n".encode() + line + b"\r\n")
                self.wfile.flush()
                time.sleep(0.2)
            done = (json.dumps({"done": True, "message": {"content": ""}}) + "\n").encode()
            self.wfile.write(f"{len(done):x}\r\n".encode() + done + b"\r\n0\r\n\r\n")
            log("COMPLETED all 100 chunks")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as exc:
            log(f"DISCONNECTED at chunk {i}: {type(exc).__name__}")


ThreadingHTTPServer(("127.0.0.1", 11999), Handler).serve_forever()
```

- [ ] **Step 2: Write the spike app** (scratchpad `spike_app.py`; replace `<REPO>` and `<SCRATCH>` with absolute paths)

```python
import sys

import streamlit as st

sys.path.insert(0, r"<REPO>")
import engine  # noqa: E402

engine.OLLAMA_CHAT_URL = "http://127.0.0.1:11999/api/chat"
LOG = r"<SCRATCH>\spike.log"


def log(msg):
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


if st.button("Start"):
    st.button("Stop")
    box = st.empty()
    buf = []

    def on_token(tok):
        buf.append(tok)
        box.markdown("".join(buf))

    try:
        engine._ollama_chat("fake", [{"role": "user", "content": "hi"}],
                            on_token=engine._guard(on_token))
        log("RETURNED normally")
    except BaseException as exc:
        log(f"RAISED {type(exc).__name__}; Exception subclass: {isinstance(exc, Exception)}")
        raise
    finally:
        log(f"FINALLY ran with {len(buf)} tokens buffered")
```

- [ ] **Step 3: Run the spike**

Start both in the background:
`python <SCRATCH>/slow_ollama.py <SCRATCH>/spike.log` and
`python -m streamlit run <SCRATCH>/spike_app.py --server.headless true --server.port 8599`.
In the built-in browser open `http://localhost:8599`, click **Start**, wait ~2 s, click **Stop**, wait 2 s, then read `spike.log`.

Expected (all three must hold):
```
RAISED RerunException; Exception subclass: False
FINALLY ran with N tokens buffered        (N ≈ 10)
DISCONNECTED at chunk M: ...              (M within a few chunks of N)
```

Stop both background processes. If any line differs, **stop here**: switch to a cooperative flag (Stop sets `st.session_state.stop_requested`; `on_token` raises a private `_Stopped(Exception)` when set — which `_guard` carries out of the chain) and re-run the spike before continuing.

- [ ] **Step 4: Implement the Stop button**

In `render_badges`, insert at the top of the function:

```python
    if meta.get("stopped"):
        st.markdown("<span class='nx-badge nx-pin'>stopped</span>", unsafe_allow_html=True)
        return
```

In `render_why`, insert at the top of the function:

```python
    if meta.get("stopped"):
        return
```

Add before `run_turn`:

```python
def _save_stopped(buffer: list[str], thoughts: list[str], opts: Options) -> None:
    """Keep what streamed before Stop was clicked.

    Runs while Streamlit unwinds the interrupted script, so it touches session
    state and the store only — no st.* calls.
    """
    partial = "".join(buffer).strip()
    if not partial and not thoughts:
        return
    meta = {
        "stopped": True,
        "thinking": "".join(thoughts) or None,
        "forced_model": bool(opts.force_model),
    }
    _persist("assistant", (partial or "_(stopped before the answer began)_") + "\n\n_— stopped_", meta)
    try:
        store.record_turn({"chat_id": st.session_state.chat_id, "stopped": True})
    except Exception as exc:
        print(f"[nexus] could not record metrics: {exc}", file=sys.stderr)
```

In `run_turn`, add the Stop button as the first thing inside `with st.chat_message("assistant"):`

```python
        stop_slot = st.empty()
        stop_slot.button("■ Stop", key="stop_generation",
                         help="Stop this answer. What's been written so far is kept.")
```

and replace the `try: … except Exception as exc: …` block with:

```python
        finished = False
        try:
            result = answer(
                question, options=opts, on_token=on_token, on_thinking=on_thinking,
                history=history, chat_id=st.session_state.chat_id,
            )
            finished = True
        except Exception as exc:
            finished = True
            stop_slot.empty()
            think_slot.empty()
            placeholder.empty()
            st.error(f"NEXUS could not answer: {exc}")
            return
        finally:
            # Clicking Stop makes Streamlit raise its rerun exception inside
            # on_token; that closes the Ollama stream on its way out.
            if not finished:
                _save_stopped(buffer, thoughts, opts)
        stop_slot.empty()
```

- [ ] **Step 5: Re-run the app tests**

Run: `PYTHONPATH=. python tests/test_app.py -v` — Expected: 4 tests OK.

- [ ] **Step 6: Manual check against a real model**

If Ollama is installed, `python run.py --check` starts it. Launch the app, ask for something long ("write a 1000-word essay about caching"), click **■ Stop** after a few lines. Confirm: the partial answer stays with a *stopped* badge, the Diagnostics "Recent answers" row shows `stopped`, and `ollama ps` / GPU usage shows generation ended within ~2 s. If Ollama isn't available, record that this check was skipped and rely on Step 3.

- [ ] **Step 7: Commit**

```bash
git add app.py
git commit -m "Add a Stop button that keeps the partial answer

A click during streaming interrupts the script inside on_token, which
closes the Ollama stream (verified against a slow NDJSON server: the
server sees the disconnect within a chunk or two). What was written is
saved with a stopped marker and recorded in the metrics."
```

---

### Task 9: Corpus fix, eval, README, final verification

**Files:**
- Modify: `documents/checklist.txt:60,63`, `program_info/checklist.txt:60,63`
- Modify: `README.md`

- [ ] **Step 1: Baseline eval**

Run: `python ingest.py` then `python eval_rag.py > "<SCRATCH>/eval_before.txt"` and read the table.

- [ ] **Step 2: Fix the stale path**

In both `documents/checklist.txt` and `program_info/checklist.txt`, replace every
`C:\Users\vashu\OneDrive\Desktop\3rd year\FML\Nexus` with
`C:\Users\vashu\OneDrive\Documents\GitHub\Nexus` (lines 60 and 63).
Verify: `grep -rn "3rd year" documents program_info` — Expected: no output.

- [ ] **Step 3: Eval after**

Run: `python ingest.py` (re-embeds only the changed file) then `python eval_rag.py > "<SCRATCH>/eval_after.txt"`.
Expected: every recall@1, recall@5, MRR and grounded@5 value ≥ its `eval_before` value. If any drops, stop and report the diff before continuing.

- [ ] **Step 4: README**

In the **Four tabs** table replace the Diagnostics row with:

```markdown
| **Diagnostics** | Installed models, what's resident in VRAM, index configuration, the last 15 routing decisions, per-answer speed (time to first token, tokens/s, cold loads), per-model medians and failure rates, and any installed models NEXUS profiled on its own. |
```

Directly after the overrides table (the paragraph ending "…one click instead of a settings change." stays below it), insert:

```markdown
**Chats are saved and follow-ups work.** Every message is stored in a local
SQLite file (`data/nexus.db`) and the sidebar lists recent chats to reopen or
delete. Earlier turns go to the model with each new question, newest first, as
many as fit. A short follow-up ("and the second one?") also borrows the previous
question for document search, so it retrieves something meaningful.

**Nothing is silently cut off.** Ollama drops the start of any prompt longer
than its context window, without an error. NEXUS asks for an 8,192-token window
explicitly and budgets every prompt into it: your question first, then
retrieved chunks in rank order, then history. If chunks had to be left out, or
the prompt still filled the window, the answer says so.

**Reasoning stays out of the answer.** Models that think before answering
(deepseek-r1, qwen3, …) stream their reasoning into a collapsed *Reasoning*
panel above the reply.

**Stop** ends an answer mid-stream and keeps what was written.
```

At the end of the **Automatic AI selection** section (after "…PC folder actions keep working regardless — they never needed a model."), add:

```markdown
**Models NEXUS has never heard of still get used.** Any installed model that
isn't in the catalogue gets a profile inferred from its name (`coder` → coding,
`r1`/`qwq` → reasoning, `vl`/`llava`/`vision` → vision, anything else general),
its size, and the capabilities Ollama reports. Inferred profiles are scored at
90%, so a hand-tuned catalogue entry wins a tie. The *Why this model* panel says
when a profile was inferred and Diagnostics lists them all; to promote one, add a
line to `CATALOG` in `providers.py`.
```

Before **Quick troubleshooting**, add:

```markdown
## Configuration

| variable | default | does |
|---|---|---|
| `NEXUS_DB` | `data/nexus.db` | where saved chats and metrics live |
| `NEXUS_NUM_CTX` | `8192` | context window requested from Ollama. Lower it if a large model spills out of VRAM |

## HTTP API (`run.py --server`)

The dependency-free server only answers its own page:

- Requests must be addressed to `127.0.0.1` or `localhost` on the server's
  port, and any `Origin` must match; otherwise `403`.
- `POST` bodies must be `application/json` (else `415`) and under 1 MB (else `413`).
- `POST /api/chat {"question": …}` never changes anything on disk. A file
  operation comes back as a preview with a `pending_id`.
- `POST /api/apply {"pending_id": …}` runs that one previewed action. Each id
  works once and expires after 10 minutes.

A page on another website can't trigger a file operation: it can't send JSON
without the server's permission, and it never sees a `pending_id`.
```

- [ ] **Step 5: Full verification**

Run: `for t in tests/test_*.py; do echo "== $t"; PYTHONPATH=. python "$t" 2>&1 | tail -3; done`
Expected: every file ends in `OK`. Then `python run.py --check` if Ollama is installed (Expected: all preflight lines pass, smoke test answers); otherwise note it was skipped.
Check commit hygiene: `git log main..HEAD --format='%an <%ae>%n%b' | grep -i -c "co-authored"` — Expected: `0`.

- [ ] **Step 6: Commit**

```bash
git add documents/checklist.txt program_info/checklist.txt README.md
git commit -m "Fix the stale project path in the indexed checklist; document Phase 1

Retrieval eval unchanged or better after re-indexing. README covers saved
chats, the context budget, reasoning, Stop, model discovery, the new
environment variables and the locked-down HTTP API."
```
