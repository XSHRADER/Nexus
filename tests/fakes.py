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
