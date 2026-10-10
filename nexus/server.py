"""
server.py
Dependency-free HTTP front end for NEXUS (`python run.py --server`).

It only serves its own page. Every request must be addressed to this server's
own host and port, POST bodies must be JSON, and a previewed file operation can
only be applied with the single-use `pending_id` issued alongside the preview —
so a page on another site cannot drive it.

The server owns each conversation: history comes from the store, not from
whatever the page sends, so a reload or a second tab can't desynchronise what
the model remembers.
"""

import base64
import binascii
import json
import logging
import secrets
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from nexus import brain, cloud, config, feedback, log, providers, store, truth_check
from nexus.config import get_settings
from nexus.engine import (
    Options,
    answer,
    apply_pending,
    arena,
    council,
    council_message_meta,
    council_recommended,
    get_router,
    message_meta,
    transcribe,
)
from nexus.router import TaskRouter

PROJECT_DIR = config.PROJECT_DIR
UI_DIR = Path(__file__).resolve().parent / "static"
HOST, PORT = "127.0.0.1", 8000
_log = logging.getLogger(__name__)
MAX_BODY = 1_000_000            # bytes, for ordinary requests
# Images and voice recordings arrive base64-encoded inside the JSON body.
MAX_UPLOAD_BODY = 25 * 1024 * 1024
UPLOAD_PATHS = ("/api/chat", "/api/arena", "/api/council")
PENDING_TTL = 600.0             # seconds a previewed action stays applicable

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


def router_status() -> dict[str, Any]:
    """What routes your questions right now, how well it measured, and how
    much of your own feedback it could learn from."""
    from nexus import learned_router

    meta = learned_router.current_meta()
    in_use = get_router().learned
    return {
        "method": f"learned {in_use.version}" if in_use else "rules",
        "meta": meta and {k: meta.get(k) for k in ("version", "created", "use_embeddings",
                                                   "heads", "metrics", "gate")},
        "feedback": feedback.summary(),
    }


def _options(payload: dict[str, Any]) -> Options:
    return Options(cloud_mode=payload.get("cloud_mode") or None,
                   allow_docs=payload.get("allow_docs"),
                   allow_paid=payload.get("allow_paid"))


def _images(payload: dict[str, Any]) -> list[dict[str, str]]:
    return [{"data": i["data"], "mime": i.get("mime") or "image/png"}
            for i in payload.get("images") or [] if isinstance(i, dict) and i.get("data")]


def _chat_for(payload: dict[str, Any], question: str) -> str:
    """The chat this request continues, or a new one titled by the question."""
    chat_id = payload.get("chat_id")
    if isinstance(chat_id, str) and store.get_chat(chat_id) is not None:
        return chat_id
    return store.create_chat(question)


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

    def _body_limit(self) -> int:
        return MAX_UPLOAD_BODY if urlparse(self.path).path in UPLOAD_PATHS else MAX_BODY

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
        if length > self._body_limit():
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
        path = parsed.path
        if path in ("/", "/index.html"):
            self.serve_file(UI_DIR / "index.html", STATIC_TYPES[".html"])
        elif path == "/api/models":
            self.send_json(TaskRouter.MODEL_MAP)
        elif path == "/api/status":
            self._status()
        elif path == "/api/chats":
            self.send_json(store.list_chats(limit=50))
        elif path.startswith("/api/chats/"):
            chat = store.get_chat(path.rstrip("/").rsplit("/", 1)[1])
            if chat is None:
                self.send_json({"error": "chat not found"}, status=404)
            else:
                self.send_json(dict(chat, messages=store.load_messages(chat["id"])))
        elif path == "/api/leaderboard":
            task = (parse_qs(parsed.query).get("task") or [None])[0] or None
            self.send_json({"task": task, "rows": feedback.leaderboard(task),
                            "summary": feedback.summary()})
        elif path == "/api/router":
            self.send_json(router_status())
        elif path == "/api/inbox":
            b = brain.get_brain()
            self.send_json({"items": b.inbox(), "unread": b.unread()})
        elif path == "/api/study":
            b = brain.get_brain()
            self.send_json({"cards": b.due_cards(), "stats": b.study_stats()})
        elif path == "/api/cloud/verify":
            self.send_json(cloud.verify_models())
        else:
            self._static(path)

    def _status(self) -> None:
        """What NEXUS can reach right now, so an "unavailable" answer makes sense."""
        settings = get_settings()
        status = providers.availability()
        status["defaults"] = {
            "cloud_mode": settings.cloud_mode,
            "allow_docs": settings.allow_docs_to_cloud,
            "allow_paid": settings.allow_paid,
        }
        status["labels"] = {p: c["label"] for p, c in cloud.PROVIDERS.items()}
        b = brain.get_brain()
        status["brain"] = {"unread": b.unread(), "due_cards": b.study_stats()["due"]}
        self.send_json(status)

    def _static(self, path: str) -> None:
        try:
            candidate = (UI_DIR / path.lstrip("/")).resolve()
            if candidate.is_file() and candidate.is_relative_to(UI_DIR):
                self.serve_file(
                    candidate,
                    STATIC_TYPES.get(candidate.suffix.lower(), "application/octet-stream"),
                )
                return
        except OSError:
            pass
        self.send_error(404, "Not found")

    def do_DELETE(self):
        if not self._same_origin():
            return
        path = urlparse(self.path).path
        if not path.startswith("/api/chats/"):
            self.send_error(404, "Not found")
            return
        chat_id = path.rstrip("/").rsplit("/", 1)[1]
        store.delete_chat(chat_id)
        self.send_json({"deleted": chat_id})

    POST_ROUTES = {
        "/api/chat": "_chat",
        "/api/apply": "_apply",
        "/api/truth": "_truth",
        "/api/feedback": "_feedback",
        "/api/arena": "_arena",
        "/api/arena/vote": "_vote",
        "/api/council": "_council",
        "/api/router/train": "_train",
        "/api/inbox/read": "_brain",
        "/api/brain/run": "_brain",
        "/api/study/answer": "_brain",
    }

    def do_POST(self):
        if not self._same_origin():
            return
        handler = self.POST_ROUTES.get(urlparse(self.path).path)
        if handler is None:
            self.send_error(404, "Not found")
            return
        payload = self._read_json()
        if payload is None:
            return
        getattr(self, handler)(payload)

    # -- chat ---------------------------------------------------------------

    def _chat(self, payload: dict[str, Any]) -> None:
        # Any "confirm" field is ignored: answering never changes the disk.
        question = str(payload.get("question") or "").strip()
        options = _options(payload)
        images = _images(payload)

        # Voice: transcribe first; the transcript becomes the question.
        transcript = None
        audio = payload.get("audio")
        if isinstance(audio, dict) and audio.get("data"):
            try:
                heard = transcribe(base64.b64decode(audio["data"], validate=True),
                                   audio.get("name") or "recording.wav",
                                   audio.get("mime") or "audio/wav", options)
            except (binascii.Error, ValueError):
                self.send_json({"error": "audio was not valid base64"}, status=400)
                return
            except RuntimeError as exc:
                self.send_json({"error": str(exc)}, status=400)
                return
            transcript = heard["text"]
            question = f"{question}\n{transcript}".strip() if question else transcript
        if not question:
            self.send_json({"error": "empty question"}, status=400)
            return

        # [council] auto = "hard": hard prompts convene the council by themselves.
        if transcript is None and council_recommended(question, options):
            self._council(dict(payload, question=question, auto=True))
            return

        chat_id = _chat_for(payload, question)
        history = store.load_messages(chat_id)
        try:
            result = answer(question, base_dir=PROJECT_DIR, history=history, options=options,
                            images=images, chat_id=chat_id)
        except Exception as exc:  # pragma: no cover - UI safety path
            self.send_json({"error": str(exc), "chat_id": chat_id}, status=500)
            return

        user_meta: dict[str, Any] = {}
        if transcript is not None:
            user_meta["voice"] = True
        if images:
            user_meta["images"] = len(images)
        store.append_message(chat_id, "user", question, user_meta or None)
        result["message_id"] = store.append_message(
            chat_id, "assistant", result.get("answer", ""), message_meta(result))
        result["chat_id"] = chat_id
        result["transcript"] = transcript
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
            result = apply_pending(pending, PROJECT_DIR)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=500)
            return
        chat_id = payload.get("chat_id")
        if isinstance(chat_id, str) and store.get_chat(chat_id) is not None:
            result["message_id"] = store.append_message(
                chat_id, "assistant", result.get("answer", ""), message_meta(result))
            result["chat_id"] = chat_id
        self.send_json(result)

    def _truth(self, payload: dict[str, Any]) -> None:
        """Check a saved answer against the documents and keep the result with it."""
        message_id = payload.get("message_id")
        message = store.get_message(message_id) if isinstance(message_id, int) else None
        if message is None or message["role"] != "assistant":
            self.send_json({"error": "answer not found"}, status=404)
            return
        meta = message.get("meta") or {}
        try:
            report = truth_check.check(message["content"], sources=meta.get("sources"))
        except Exception as exc:  # pragma: no cover - UI safety path
            self.send_json({"error": f"truth check failed: {exc}"}, status=500)
            return
        store.update_meta(message_id, {"truth": report})
        self.send_json(report)

    def _feedback(self, payload: dict[str, Any]) -> None:
        try:
            out = feedback.rate(int(payload.get("message_id")), int(payload.get("rating", 0)),
                                payload.get("reason"))
        except (TypeError, ValueError) as exc:
            self.send_json({"error": str(exc)}, status=400)
            return
        self.send_json(out)

    # -- Arena and council ----------------------------------------------------

    def _arena(self, payload: dict[str, Any]) -> None:
        """Two blind answers. Model names stay on the server until the vote."""
        question = str(payload.get("question") or "").strip()
        if not question:
            self.send_json({"error": "empty question"}, status=400)
            return
        chat_id = _chat_for(payload, question)
        try:
            out = arena(question, options=_options(payload), history=store.load_messages(chat_id),
                        images=_images(payload), chat_id=chat_id)
        except Exception as exc:  # pragma: no cover - UI safety path
            self.send_json({"error": str(exc), "chat_id": chat_id}, status=500)
            return
        if out.get("error"):
            self.send_json({"error": out["error"], "chat_id": chat_id}, status=400)
            return
        self.send_json({
            "battle_id": out["battle_id"], "chat_id": chat_id, "task": out["task"],
            "note": out["note"],
            "a": {"answer": out["a"]["answer"]}, "b": {"answer": out["b"]["answer"]},
        })

    def _vote(self, payload: dict[str, Any]) -> None:
        """Record the vote, reveal the models, and continue the chat with the
        preferred answer (A on a tie)."""
        try:
            battle = feedback.vote(int(payload.get("battle_id")), str(payload.get("winner")))
        except (TypeError, ValueError) as exc:
            self.send_json({"error": str(exc)}, status=400)
            return
        chat_id = battle["chat_id"]
        message_id = None
        if chat_id is not None and store.get_chat(chat_id) is not None:
            text, meta = feedback.battle_message(battle)
            store.append_message(chat_id, "user", battle["prompt"], {"arena": battle["id"]})
            message_id = store.append_message(chat_id, "assistant", text, meta)
        self.send_json({
            "battle_id": battle["id"], "winner": battle["winner"], "chat_id": chat_id,
            "message_id": message_id,
            "a": {"model": battle["model_a"], "provider": battle["provider_a"]},
            "b": {"model": battle["model_b"], "provider": battle["provider_b"]},
        })

    def _council(self, payload: dict[str, Any]) -> None:
        """Several models answer, a judge merges them; the merged answer
        joins the chat with the whole council kept in its metadata."""
        question = str(payload.get("question") or "").strip()
        if not question:
            self.send_json({"error": "empty question"}, status=400)
            return
        chat_id = _chat_for(payload, question)
        images = _images(payload)
        try:
            out = council(question, options=_options(payload),
                          history=store.load_messages(chat_id), images=images, chat_id=chat_id)
        except Exception as exc:  # pragma: no cover - UI safety path
            self.send_json({"error": str(exc), "chat_id": chat_id}, status=500)
            return
        if out.get("error"):
            self.send_json({"error": out["error"], "chat_id": chat_id}, status=400)
            return
        store.append_message(chat_id, "user", question,
                             {"council": True, "images": len(images) or None})
        meta = council_message_meta(out, auto=bool(payload.get("auto")))
        message_id = store.append_message(chat_id, "assistant", out["answer"], meta)
        self.send_json(dict(meta, answer=out["answer"], chat_id=chat_id, message_id=message_id))

    # -- learned router and background brain -----------------------------------

    def _train(self, _payload: dict[str, Any]) -> None:
        """Retrain from seed data + your feedback (seconds), then use the new
        router straight away if it passed its gate."""
        from train.train_router import main as train_main

        try:
            report = train_main([])
        except Exception as exc:  # pragma: no cover - UI safety path
            self.send_json({"error": f"training failed: {exc}"}, status=500)
            return
        get_router().reload_learned()
        self.send_json(dict(router_status(), last_training={
            "version": report["version"], "made_current": report["made_current"],
            "gate": report["gate"], "golden": report["metrics"]["golden"]}))

    def _brain(self, payload: dict[str, Any]) -> None:
        path = urlparse(self.path).path
        b = brain.get_brain()
        try:
            if path == "/api/inbox/read":
                item = payload.get("id")
                b.mark_read(int(item) if item is not None else None)
                self.send_json({"unread": b.unread()})
            elif path == "/api/study/answer":
                out = b.review(int(payload["card_id"]), bool(payload.get("knew")))
                self.send_json(dict(out, stats=b.study_stats()))
            else:
                action = payload.get("action")
                if action == "digest":
                    out = {"digest": b.make_digest()}
                elif action == "study_all":
                    out = {"cards": b.study_all()}
                else:
                    out = b.tick()
                self.send_json(dict(out, unread=b.unread(), stats=b.study_stats()))
        except (KeyError, TypeError, ValueError) as exc:
            self.send_json({"error": str(exc)}, status=400)

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
        limit = self._body_limit()
        if 0 < length <= limit:
            self.rfile.read(length)
        elif length > limit:
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
    if get_settings().brain_enabled:
        brain.get_brain().start()
        print("Background brain: watching documents/ for changes")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping NEXUS UI...")
    finally:
        server.server_close()
