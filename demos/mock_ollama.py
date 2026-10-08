"""
demos/mock_ollama.py
A stand-in for Ollama, for demos and testing on machines without it.

It speaks the parts of Ollama's HTTP API that NEXUS uses (/api/tags, /api/ps,
/api/chat, streaming or not) and answers from a few canned facts. It is not a
model: its one job is to make it visible *what NEXUS sent*. Every reply ends
with how many messages it received, so memory working (or not) shows up in
the answer itself.

    python demos/mock_ollama.py            # listens on 127.0.0.1:11434
    python demos/mock_ollama.py --port 11500

Stop real Ollama first if it's running -- both want port 11434.
"""

import argparse
import json
import re
import time
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mock_judge  # noqa: E402

MODELS = ["llama3.1:8b", "qwen2.5-coder:7b", "deepseek-r1:7b"]

FACTS = {
    "france": ("Paris", "about 2.1 million people (city proper)"),
    "japan": ("Tokyo", "about 14 million people"),
    "india": ("New Delhi", "about 250,000 people (the NDMC area; Delhi as a whole is over 30 million)"),
}
CITY_POPULATION = {city.lower(): pop for city, pop in FACTS.values()}


# Each mock model answers an "explain ..." question in its own style, so a
# blind Arena comparison has something to compare.
STYLES = {
    "llama3.1:8b": "{Topic} is easiest to picture with an everyday example. Think of it as a "
                   "labelled set of drawers: you know the label, so you open the right drawer "
                   "straight away instead of searching them all.",
    "deepseek-r1:7b": "Let me reason it through. {Topic} works in three steps: (1) a function maps "
                      "each key to a position, (2) the value is stored at that position, and "
                      "(3) clashes between keys are resolved by chaining or probing. On average "
                      "a lookup takes constant time; in the worst case it degrades to linear.",
    "qwen2.5-coder:7b": "Short version: {topic}: key -> hash(key) -> slot. In Python it's a dict: "
                        "`d = {{'a': 1}}; d['a']`.",
}


def mock_cards(passage: str) -> str:
    """Flashcards the way a model might write them -- including one with a
    wrong number, which the truth check should throw away."""
    cards = []
    for sentence in re.split(r"(?<=[.!?])\s+", passage):
        m = re.match(r"(.{3,60}?)\s+(is|are|uses|provides|stores|holds)\s+(.+?)[.!?]?$", sentence.strip())
        if m:
            cards.append({"q": f"{m.group(1)} {m.group(2)} what?", "a": sentence.strip()})
    numbered = next((c for c in cards if re.search(r"\d+", c["a"])), None)
    if numbered:
        wrong = re.sub(r"\d+", lambda d: str(int(d.group(0)) + 7), numbered["a"], count=1)
        cards.append({"q": numbered["q"] + " (check)", "a": wrong})
    return json.dumps(cards[:4])


def mock_digest(notes: str) -> str:
    bullets = []
    for block in re.split(r"\n\n(?=\[)", notes):
        m = re.match(r"\[([^\]]+)\]\n(.*)", block, re.S)
        if m:
            body = re.sub(r"^#+ .*$", "", m.group(2), flags=re.M)  # skip headings
            first = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", body).strip())[0]
            bullets.append(f"- {m.group(1)}: {first[:160]}")
    return "\n".join(bullets) or "- Nothing notable."


def reply_for(messages: list[dict], model: str = "") -> str:
    system = " ".join(m.get("content") or "" for m in messages if m.get("role") == "system")
    if mock_judge.MARKER in system:
        return mock_judge.verdict(messages[-1].get("content") or "")
    if "Write flashcards" in system:
        return mock_cards(messages[-1].get("content") or "")
    if "weekly digest" in system:
        return mock_digest(messages[-1].get("content") or "")
    convo = [m for m in messages if m.get("role") in ("user", "assistant")]
    question = convo[-1]["content"].lower() if convo else ""

    country = next((c for c in FACTS if c in question), None)
    if "capital" in question and country:
        text = f"The capital of {country.title()} is {FACTS[country][0]}."
    elif "population" in question or "how many people" in question:
        # Resolve "its" from the conversation, newest mention first. With no
        # history there is nothing to resolve it against.
        city = None
        for m in reversed(convo):
            words = m["content"].lower()
            city = next((c for c in CITY_POPULATION if re.search(rf"\b{c}\b", words)), None)
            if city:
                break
        if city:
            text = f"{city.title()} has {CITY_POPULATION[city]}."
        else:
            text = "Which place do you mean? I don't see one earlier in our conversation."
    elif re.match(r"^\s*(explain|what is|what's)\b", question) and model in STYLES:
        topic = re.sub(r"^\s*(explain|what is|what's)\s+(a |an |the )?", "", question).strip(" ?.")
        text = STYLES[model].format(topic=topic, Topic=topic[:1].upper() + topic[1:])
    elif "nexus" in question and re.search(r"\b(document|tell me|about|explain|how)", question):
        # Deliberately mixed, so the truth check has something to find:
        # true, true, wrong (the port is 11434), true, and unknowable.
        text = ("Here is what NEXUS is built from:\n\n"
                "- NEXUS stores its document vectors in Chroma.\n"
                "- It embeds documents with the all-MiniLM-L6-v2 model.\n"
                "- Ollama serves the models at localhost:8080.\n"
                "- Supported formats are TXT, Markdown, PDF and DOCX.\n"
                "- NEXUS was first released in 2019 by a team at Google.")
    elif re.search(r"\b(hi|hello|hey)\b", question):
        text = "Hello! Ask me something, then ask a follow-up."
    else:
        text = "I'm a mock model, so I only know a few canned facts."

    return f"{text}\n\n_(mock model · received {len(messages)} message{'s' if len(messages) != 1 else ''})_"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # keep the demo output clean
        pass

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            self._json({"models": [{"name": m, "model": m} for m in MODELS]})
        elif self.path == "/api/ps":
            self._json({"models": [{"name": MODELS[0], "model": MODELS[0]}]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/api/chat":
            self._json({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(length) or b"{}")
        model = req.get("model", "")
        if model not in MODELS:
            self._json({"error": f"model '{model}' not found"}, 404)
            return
        text = reply_for(req.get("messages") or [], model)

        if not req.get("stream", True):
            self._json({"model": model, "message": {"role": "assistant", "content": text},
                        "done": True})
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for word in re.findall(r"\S+\s*", text):
            line = {"model": model, "message": {"role": "assistant", "content": word},
                    "done": False}
            self.wfile.write((json.dumps(line) + "\n").encode())
            self.wfile.flush()
            time.sleep(0.02)
        self.wfile.write((json.dumps({"model": model, "done": True}) + "\n").encode())


def start_in_thread(port: int = 0):
    """Start on `port` (0 = any free port). Returns (server, base_url)."""
    import threading

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=11434)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"mock Ollama on http://127.0.0.1:{args.port} · models: {', '.join(MODELS)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
