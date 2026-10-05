"""
server.py
Dependency-free HTTP front end for NEXUS (`python run.py --server`).

It only serves its own page. Every request must be addressed to this server's
own host and port, POST bodies must be JSON, and a previewed file operation can
only be applied with the single-use `pending_id` issued alongside the preview —
so a page on another site cannot drive it.
"""

import json
import logging
import secrets
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from nexus import config, log, providers
from nexus.engine import answer, apply_pending
from nexus.router import TaskRouter

PROJECT_DIR = config.PROJECT_DIR
UI_DIR = Path(__file__).resolve().parent / "static"
HOST, PORT = "127.0.0.1", 8000
_log = logging.getLogger(__name__)
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
        self._body_consumed = True
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

    def _drain_body(self) -> None:
        """Read a POST body we're about to reject without using.

        Closing a socket that still holds unread bytes makes the OS reset the
        connection, and the client never sees the 403/413/415. An oversized
        body is only drained as far as it has already arrived — bounded in
        size and by a short idle timeout — so it can't tie the server up.
        """
        if self.command != "POST" or getattr(self, "_body_consumed", False):
            return
        self._body_consumed = True
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return
        if 0 < length <= MAX_BODY:
            self.rfile.read(length)
        elif length > MAX_BODY:
            self.connection.settimeout(0.2)
            drained = 0
            try:
                while drained < 4 * MAX_BODY:
                    chunk = self.rfile.read1(65536)
                    if not chunk:
                        break
                    drained += len(chunk)
            except OSError:
                pass  # idle timeout: nothing more has arrived

    def log_message(self, format, *args):
        """Request lines go to the log file, not the console."""
        _log.debug("%s %s", self.address_string(), format % args)

    def send_json(self, payload, status=200):
        self._drain_body()
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
    log.setup()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"NEXUS UI running at http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping NEXUS UI...")
    finally:
        server.server_close()
