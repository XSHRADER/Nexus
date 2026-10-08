"""
demos/mock_cloud.py
A stand-in for the cloud providers, for demos and tests without API keys or
internet. Not a model: like mock_ollama.py, its replies say what it received.

Each provider lives under its own path prefix, so one server plays all of
them. Point NEXUS at it in nexus.toml:

    [cloud.providers.gemini]
    base_url = "http://127.0.0.1:11500/gemini/v1"

Speaks:
  OpenAI-style  POST /<provider>/v1/chat/completions   (JSON or SSE stream)
                GET  /<provider>/v1/models
                POST /<provider>/v1/audio/transcriptions

Failures on demand, to show fallbacks:
    POST /_control  {"fail": {"groq": 429, "gemini": 500}}   ({} clears)
A key of "bad-key" always gets a 401.

    python demos/mock_cloud.py [--port 11500]
"""

import argparse
import json
import re
import threading
import time
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mock_judge  # noqa: E402

MODELS = {
    "gemini": ["gemini-3.7-flash"],
    "groq": ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "whisper-large-v3-turbo"],
    "openrouter": ["openrouter/auto"],
    "deepseek": ["deepseek-reasoner", "deepseek-chat"],
    "mistral": ["mistral-large-latest"],
}

# What the mock "hears" in any recording.
TRANSCRIPT = "What is the capital of Japan?"

STATE = {"fail": {}, "requests": []}
_lock = threading.Lock()


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def _count_images(messages) -> int:
    n = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            n += sum(1 for p in content if isinstance(p, dict)
                     and p.get("type") == "image_url")
    return n


def teacher_label(prompt: str) -> str:
    """A crude stand-in for a teacher model labelling a prompt (train/teacher_label.py)."""
    p = prompt.lower()
    rules = [("system_agent", r"\b(sort|organi[sz]e|tidy|duplicate|declutter)\b"),
             ("vision", r"\b(image|photo|picture|screenshot|chart)\b"),
             ("speech", r"\b(audio|recording|voice|transcribe)\b"),
             ("coding", r"\b(code|function|error|bug|python|sql|script)\b"),
             ("planning", r"\b(plan|roadmap|schedule|step by step|milestones?)\b"),
             ("reasoning", r"\b(compare|trade-?offs?|pros|cons|should i|why)\b")]
    task = next((t for t, rx in rules if re.search(rx, p)), "general")
    docs = bool(re.search(r"\b(my notes|my documents|according to|readme)\b", p))
    return json.dumps({"task": task, "needs_docs": docs})


def compose_reply(provider: str, model: str, messages: list) -> str:
    convo = [m for m in messages if m.get("role") in ("user", "assistant")]
    question = _text_of(convo[-1]["content"]) if convo else ""
    instructions = " ".join(_text_of(m.get("content")) for m in messages
                                     if m.get("role") == "system")
    if mock_judge.MARKER in instructions:
        return mock_judge.verdict(question)
    if "Reply with JSON only" in instructions:
        inner = re.search(r"<<<\n(.*)\n>>>", question, re.S)
        return teacher_label(inner.group(1) if inner else question)
    images = _count_images(messages)
    if images:
        body = (f"I can see {images} image{'s' if images != 1 else ''} you attached. "
                "(A real vision model would describe it here.)")
    elif re.search(r"\b(plan|roadmap|step[- ]by[- ]step|milestones?)\b", question, re.I):
        body = ("Here is a plan:\n1. Define the goal and success measure.\n"
                "2. Break the work into weekly milestones.\n"
                "3. Review progress each Friday and adjust.")
    else:
        body = f"You asked: “{question[:120]}”"
    sys_note = " · had a system prompt" if any(
        m.get("role") == "system" for m in messages) else ""
    return (f"{body}\n\n_(mock {provider} cloud · {model} · received "
            f"{len(messages)} message{'s' if len(messages) != 1 else ''}{sys_note})_")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    # -- helpers -----------------------------------------------------------------

    def _send(self, status: int, payload, headers: dict | None = None):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(length) if length else b""

    def _route(self):
        m = re.match(r"^/([a-z]+)(/.*)$", self.path.split("?")[0])
        return (m.group(1), m.group(2)) if m else (None, None)

    def _key(self, provider: str) -> str:
        auth = self.headers.get("Authorization", "")
        return auth[7:] if auth.startswith("Bearer ") else ""

    def _check(self, provider: str) -> bool:
        """Auth and injected failures. Returns False if a response was sent."""
        key = self._key(provider)
        if not key or key == "bad-key":
            self._send(401, {"error": {"message": "Invalid API key"}})
            return False
        status = STATE["fail"].get(provider)
        if status:
            headers = {"retry-after": "30"} if status == 429 else {}
            self._send(status, {"error": {"message": f"mock {status}"}}, headers)
            return False
        return True

    # -- verbs -------------------------------------------------------------------

    def do_GET(self):
        provider, rest = self._route()
        if provider in MODELS and rest == "/v1/models":
            if not self._check(provider):
                return
            self._send(200, {"object": "list", "data": [{"id": m, "object": "model"}
                                                        for m in MODELS[provider]]})
            return
        self._send(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if self.path == "/_control":
            cmd = json.loads(self._body() or b"{}")
            with _lock:
                if "fail" in cmd:
                    STATE["fail"] = {k: int(v) for k, v in cmd["fail"].items()}
            self._send(200, {"fail": STATE["fail"]})
            return

        provider, rest = self._route()
        if provider not in MODELS:
            self._send(404, {"error": {"message": "not found"}})
            return
        if rest == "/v1/chat/completions":
            self._openai_chat(provider)
        elif rest == "/v1/audio/transcriptions":
            self._transcribe(provider)
        else:
            self._send(404, {"error": {"message": "not found"}})

    # -- OpenAI-style ------------------------------------------------------------

    def _openai_chat(self, provider):
        body = self._body()
        if not self._check(provider):
            return
        req = json.loads(body or b"{}")
        model = req.get("model", "")
        with _lock:
            STATE["requests"].append({"provider": provider, "body": req})
        if model not in MODELS[provider]:
            self._send(404, {"error": {"message": f"model {model} not found"}})
            return
        text = compose_reply(provider, model, req.get("messages") or [])
        if not req.get("stream"):
            self._send(200, {"id": "mock", "object": "chat.completion", "model": model,
                             "choices": [{"index": 0, "finish_reason": "stop",
                                          "message": {"role": "assistant", "content": text}}]})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(b": keep-alive\n\n")
        for word in re.findall(r"\S+\s*", text):
            chunk = {"choices": [{"index": 0, "delta": {"content": word}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.01)
        self.wfile.write(b"data: [DONE]\n\n")
        self.close_connection = True

    def _transcribe(self, provider):
        self._body()  # multipart; contents don't matter to the mock
        if not self._check(provider):
            return
        self._send(200, {"text": TRANSCRIPT})


def start_in_thread(port: int = 0):
    """Start on `port` (0 = any free port). Returns (server, base_url)."""
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def reset():
    with _lock:
        STATE["fail"] = {}
        STATE["requests"].clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=11500)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"mock cloud on http://127.0.0.1:{args.port} · providers: {', '.join(MODELS)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
