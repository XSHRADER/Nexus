import base64
import binascii
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import brain
import cloud
import feedback
import providers
import truth_check
from config import get_settings
from engine import (Options, answer, arena, council, council_message_meta,
                    council_recommended, get_router, transcribe)
from store import get_store

# Base64 image/audio uploads arrive inside the JSON body.
MAX_BODY_BYTES = 25 * 1024 * 1024

PROJECT_DIR = Path(__file__).resolve().parent
UI_DIR = PROJECT_DIR / "ui"
router = get_router()


def router_status() -> dict:
    """What routes your questions right now, how well it measured, and how
    much of your own feedback it could learn from."""
    import learned_router

    meta = learned_router.current_meta()
    in_use = get_router().learned
    return {
        "method": f"learned {in_use.version}" if in_use else "rules",
        "meta": meta and {k: meta.get(k) for k in ("version", "created", "use_embeddings",
                                                   "heads", "metrics", "gate")},
        "feedback": feedback.summary(),
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.serve_file(UI_DIR / "index.html", "text/html; charset=utf-8")
            return
        if parsed.path == "/api/models":
            self.send_json(router.MODEL_MAP)
            return
        if parsed.path == "/api/chats":
            self.send_json(get_store().list_chats())
            return
        if parsed.path.startswith("/api/chats/"):
            chat_id = self._chat_id(parsed.path)
            chat = get_store().get_chat(chat_id) if chat_id is not None else None
            if chat is None:
                self.send_json({"error": "chat not found"}, status=404)
                return
            self.send_json(dict(chat, messages=get_store().get_messages(chat_id)))
            return
        if parsed.path == "/api/status":
            # What NEXUS can reach right now — the UI shows this so the user
            # can see why a given AI answered, without ever choosing one.
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
            return
        if parsed.path == "/api/leaderboard":
            task = (parse_qs(parsed.query).get("task") or [None])[0] or None
            self.send_json({"task": task, "rows": feedback.leaderboard(task),
                            "summary": feedback.summary()})
            return
        if parsed.path == "/api/router":
            self.send_json(router_status())
            return
        if parsed.path == "/api/inbox":
            b = brain.get_brain()
            self.send_json({"items": b.inbox(), "unread": b.unread()})
            return
        if parsed.path == "/api/study":
            b = brain.get_brain()
            self.send_json({"cards": b.due_cards(), "stats": b.study_stats()})
            return
        if parsed.path == "/api/cloud/verify":
            self.send_json(cloud.verify_models())
            return
        if parsed.path.startswith("/"):
            try:
                candidate = (UI_DIR / parsed.path.lstrip("/")).resolve()
                if candidate.exists() and candidate.is_file() and candidate.is_relative_to(UI_DIR):
                    suffix = candidate.suffix.lower()
                    mime = {
                        ".html": "text/html; charset=utf-8",
                        ".css": "text/css; charset=utf-8",
                        ".js": "application/javascript; charset=utf-8",
                        ".json": "application/json; charset=utf-8",
                    }.get(suffix, "application/octet-stream")
                    self.serve_file(candidate, mime)
                    return
            except OSError:
                pass
        self.send_error(404, "Not found")

    @staticmethod
    def _chat_id(path: str) -> int | None:
        try:
            return int(path.rstrip("/").rsplit("/", 1)[1])
        except (IndexError, ValueError):
            return None

    def do_DELETE(self):
        parsed = urlparse(self.path)
        chat_id = self._chat_id(parsed.path) if parsed.path.startswith("/api/chats/") else None
        if chat_id is None:
            self.send_error(404, "Not found")
            return
        get_store().delete_chat(chat_id)
        self.send_json({"deleted": chat_id})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/truth":
            self.handle_truth()
            return
        if parsed.path == "/api/feedback":
            self.handle_feedback()
            return
        if parsed.path == "/api/arena":
            self.handle_arena()
            return
        if parsed.path == "/api/arena/vote":
            self.handle_vote()
            return
        if parsed.path == "/api/router/train":
            self.handle_train()
            return
        if parsed.path == "/api/council":
            self.handle_council(self.read_json())
            return
        if parsed.path in ("/api/inbox/read", "/api/brain/run", "/api/study/answer"):
            self.handle_brain(parsed.path, self.read_json())
            return
        if parsed.path != "/api/chat":
            self.send_error(404, "Not found")
            return

        length = int(self.headers.get("Content-Length", "0"))
        if length > MAX_BODY_BYTES:
            self.send_json({"error": "attachment too large (25 MB max)"}, status=413)
            return
        data = self.rfile.read(length)
        payload = json.loads(data.decode("utf-8") or "{}")
        question = (payload.get("question") or "").strip()
        confirm = bool(payload.get("confirm", False))
        options = Options(
            cloud_mode=payload.get("cloud_mode") or None,
            allow_docs=payload.get("allow_docs"),
            allow_paid=payload.get("allow_paid"),
        )
        images = [
            {"data": img["data"], "mime": img.get("mime") or "image/png"}
            for img in payload.get("images") or []
            if isinstance(img, dict) and img.get("data")
        ]

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
        if not confirm and transcript is None and council_recommended(question, options):
            self.handle_council(dict(payload, question=question, auto=True))
            return

        # The server owns the conversation: history comes from the store, not
        # from whatever the page sends, so a reload or a second tab can't
        # desynchronise what the model remembers.
        store = get_store()
        chat_id = payload.get("chat_id")
        if not isinstance(chat_id, int) or store.get_chat(chat_id) is None:
            chat_id = store.create_chat(question)
        history = store.get_messages(chat_id)

        try:
            result = answer(
                question, base_dir=PROJECT_DIR, confirm=confirm, history=history,
                options=options, images=images,
            )
        except Exception as exc:  # pragma: no cover - UI safety path
            self.send_json({"error": str(exc), "chat_id": chat_id}, status=500)
            return

        # A confirmation re-sends the same question; only its outcome is new.
        if not confirm:
            user_meta = {}
            if transcript is not None:
                user_meta["voice"] = True
            if images:
                user_meta["images"] = len(images)
            store.add_message(chat_id, "user", question, user_meta or None)
        result["message_id"] = store.add_message(
            chat_id, "assistant", result.get("answer", ""),
            {k: result.get(k) for k in ("model", "provider", "task", "complexity",
                                       "needs_rag", "why", "info", "history_used",
                                       "local", "sources", "router_method")},
        )
        result["chat_id"] = chat_id
        result["transcript"] = transcript
        self.send_json(result)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length > MAX_BODY_BYTES:
            raise ValueError("request too large")
        return json.loads(self.rfile.read(length).decode("utf-8") or "{}")

    @staticmethod
    def options_from(payload: dict) -> Options:
        return Options(cloud_mode=payload.get("cloud_mode") or None,
                       allow_docs=payload.get("allow_docs"),
                       allow_paid=payload.get("allow_paid"))

    def handle_feedback(self):
        payload = self.read_json()
        try:
            out = feedback.rate(int(payload.get("message_id")), int(payload.get("rating", 0)),
                                payload.get("reason"))
        except (TypeError, ValueError) as exc:
            self.send_json({"error": str(exc)}, status=400)
            return
        self.send_json(out)

    def handle_arena(self):
        """Two blind answers. Model names stay on the server until the vote."""
        payload = self.read_json()
        question = (payload.get("question") or "").strip()
        if not question:
            self.send_json({"error": "empty question"}, status=400)
            return
        store = get_store()
        chat_id = payload.get("chat_id")
        if not isinstance(chat_id, int) or store.get_chat(chat_id) is None:
            chat_id = store.create_chat(question)
        images = [{"data": i["data"], "mime": i.get("mime") or "image/png"}
                  for i in payload.get("images") or [] if isinstance(i, dict) and i.get("data")]
        try:
            out = arena(question, options=self.options_from(payload),
                        history=store.get_messages(chat_id), images=images, chat_id=chat_id)
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

    def handle_vote(self):
        """Record the vote, reveal the models, and continue the chat with the
        preferred answer (A on a tie)."""
        payload = self.read_json()
        try:
            battle = feedback.vote(int(payload.get("battle_id")), str(payload.get("winner")))
        except (TypeError, ValueError) as exc:
            self.send_json({"error": str(exc)}, status=400)
            return
        store = get_store()
        chat_id = battle["chat_id"]
        message_id = None
        if chat_id is not None and store.get_chat(chat_id) is not None:
            store.add_message(chat_id, "user", battle["prompt"], {"arena": battle["id"]})
            if battle["winner"] == "both_bad":
                text, side = "(Arena: both answers were marked bad.)", None
            else:
                side = "b" if battle["winner"] == "b" else "a"
                text = battle[f"answer_{side}"] or ""
            meta = {"task": battle["task"], "complexity": battle["complexity"],
                    "arena": {"battle_id": battle["id"], "winner": battle["winner"],
                              "a": battle["model_a"], "b": battle["model_b"]}}
            if side:
                model = battle[f"model_{side}"]
                spec = providers.spec_by_name(model)
                meta.update(model=model, provider=battle[f"provider_{side}"],
                            local=spec.is_local if spec else True)
            message_id = store.add_message(chat_id, "assistant", text, meta)
        self.send_json({
            "battle_id": battle["id"], "winner": battle["winner"], "chat_id": chat_id,
            "message_id": message_id,
            "a": {"model": battle["model_a"], "provider": battle["provider_a"]},
            "b": {"model": battle["model_b"], "provider": battle["provider_b"]},
        })

    def handle_council(self, payload: dict):
        """Several models answer, a judge merges them; the merged answer
        joins the chat with the whole council kept in its metadata."""
        question = (payload.get("question") or "").strip()
        if not question:
            self.send_json({"error": "empty question"}, status=400)
            return
        store = get_store()
        chat_id = payload.get("chat_id")
        if not isinstance(chat_id, int) or store.get_chat(chat_id) is None:
            chat_id = store.create_chat(question)
        images = [{"data": i["data"], "mime": i.get("mime") or "image/png"}
                  for i in payload.get("images") or [] if isinstance(i, dict) and i.get("data")]
        try:
            out = council(question, options=self.options_from(payload),
                          history=store.get_messages(chat_id), images=images)
        except Exception as exc:  # pragma: no cover - UI safety path
            self.send_json({"error": str(exc), "chat_id": chat_id}, status=500)
            return
        if out.get("error"):
            self.send_json({"error": out["error"], "chat_id": chat_id}, status=400)
            return
        store.add_message(chat_id, "user", question, {"council": True, "images": len(images) or None})
        meta = council_message_meta(out, auto=bool(payload.get("auto")))
        message_id = store.add_message(chat_id, "assistant", out["answer"], meta)
        self.send_json(dict(meta, answer=out["answer"], chat_id=chat_id, message_id=message_id))

    def handle_brain(self, path: str, payload: dict):
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

    def handle_train(self):
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

    def handle_truth(self):
        """Check a saved answer against the documents and keep the result with it."""
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        message_id = payload.get("message_id")
        message = get_store().get_message(message_id) if isinstance(message_id, int) else None
        if message is None or message["role"] != "assistant":
            self.send_json({"error": "answer not found"}, status=404)
            return
        meta = message.get("meta") or {}
        try:
            report = truth_check.check(message["content"], sources=meta.get("sources"))
        except Exception as exc:  # pragma: no cover - UI safety path
            self.send_json({"error": f"truth check failed: {exc}"}, status=500)
            return
        get_store().update_meta(message_id, {"truth": report})
        self.send_json(report)

    def send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
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
    if get_settings().brain_enabled:
        brain.get_brain().start()
        print("Background brain: watching documents/ for changes")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping NEXUS UI...")
    finally:
        server.server_close()
