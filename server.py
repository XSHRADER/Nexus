import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import providers
from engine import answer, get_router

PROJECT_DIR = Path(__file__).resolve().parent
UI_DIR = PROJECT_DIR / "ui"
router = get_router()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.serve_file(UI_DIR / "index.html", "text/html; charset=utf-8")
            return
        if parsed.path == "/api/models":
            self.send_json(router.MODEL_MAP)
            return
        if parsed.path == "/api/status":
            # What NEXUS can reach right now — the UI shows this so the user
            # can see why a given AI answered, without ever choosing one.
            self.send_json(providers.availability())
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

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/chat":
            self.send_error(404, "Not found")
            return

        length = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(length)
        payload = json.loads(data.decode("utf-8") or "{}")
        question = payload.get("question", "")
        confirm = bool(payload.get("confirm", False))

        try:
            result = answer(question, base_dir=PROJECT_DIR, confirm=confirm)
            self.send_json(result)
        except Exception as exc:  # pragma: no cover - UI safety path
            self.send_json({"error": str(exc)}, status=500)

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
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping NEXUS UI...")
    finally:
        server.server_close()
