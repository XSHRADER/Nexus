"""Conversation memory, saved chats and config (Phase 0).

No Ollama, no network and no embedding model: the router and HTTP calls are
replaced with fakes, so these run anywhere.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config
import engine
from store import ChatStore, make_title


def _decision(task="general", chain=None, needs_rag=False):
    return {
        "task": task,
        "model": "llama3.1:8b",
        "provider": "ollama",
        "chain": chain if chain is not None else [
            {"model": "llama3.1:8b", "provider": "ollama", "score": 1.0, "reason": "test"}
        ],
        "complexity": 0.1,
        "available": {"ollama": ["llama3.1:8b"]},
        "needs_rag": needs_rag,
        "confidence": 1.0,
        "scores": {"general": 1.0},
        "reason": "test",
    }


class FakeRouter:
    def __init__(self, decision):
        self.decision = decision

    def route(self, question, force_task=None, **kwargs):
        self.kwargs = kwargs
        return self.decision


class FakeResponse:
    def __init__(self, body=None, lines=None):
        self._body = body or {}
        self._lines = lines or []

    def raise_for_status(self):
        pass

    def json(self):
        return self._body

    def iter_lines(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TrimHistoryTests(unittest.TestCase):
    def _turns(self, n):
        return [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
            for i in range(n)
        ]

    def test_keeps_newest_in_order(self):
        kept = engine.trim_history(self._turns(10), max_messages=4, max_chars=10_000)
        self.assertEqual([m["content"] for m in kept], ["m6", "m7", "m8", "m9"])

    def test_char_budget_drops_oldest_whole_messages(self):
        history = [
            {"role": "user", "content": "a" * 50},
            {"role": "assistant", "content": "b" * 50},
            {"role": "user", "content": "c" * 50},
            {"role": "assistant", "content": "d" * 50},
        ]
        kept = engine.trim_history(history, max_messages=10, max_chars=120)
        # Two newest fit (100 chars); a third would exceed 120. The window
        # then can't open on an assistant turn, so it starts at "c".
        self.assertEqual([m["content"][0] for m in kept], ["c", "d"])

    def test_never_starts_with_assistant(self):
        kept = engine.trim_history(self._turns(5), max_messages=2, max_chars=10_000)
        self.assertEqual(kept[0]["role"], "user")

    def test_drops_metadata_bad_roles_and_blanks(self):
        history = [
            {"role": "user", "content": "q", "meta": {"x": 1}, "id": 3},
            {"role": "system", "content": "ignore me"},
            {"role": "assistant", "content": "   "},
            {"role": "assistant", "content": "a"},
            "not a dict",
        ]
        kept = engine.trim_history(history, max_messages=10, max_chars=10_000)
        self.assertEqual(kept, [{"role": "user", "content": "q"},
                                {"role": "assistant", "content": "a"}])

    def test_zero_budget_means_no_memory(self):
        self.assertEqual(engine.trim_history(self._turns(4), max_messages=0, max_chars=100), [])

    def test_none_history(self):
        self.assertEqual(engine.trim_history(None, max_messages=4, max_chars=100), [])


class BuildMessagesTests(unittest.TestCase):
    def test_order_system_history_question(self):
        msgs = engine.build_messages(
            "and now?", [{"role": "user", "content": "q1"},
                         {"role": "assistant", "content": "a1"}], system="ctx")
        self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant", "user"])
        self.assertEqual(msgs[-1]["content"], "and now?")

    def test_no_system_when_none(self):
        msgs = engine.build_messages("hi")
        self.assertEqual(msgs, [{"role": "user", "content": "hi"}])

    def test_short_follow_up_retrieves_with_previous_question(self):
        history = [{"role": "user", "content": "Which vector store does NEXUS use?"},
                   {"role": "assistant", "content": "Chroma."}]
        q = engine.retrieval_query("why that one?", history)
        self.assertIn("Which vector store", q)
        self.assertIn("why that one?", q)

    def test_long_question_retrieves_alone(self):
        history = [{"role": "user", "content": "earlier"}]
        q = "explain how the hybrid retrieval pipeline fuses vector and keyword results"
        self.assertEqual(engine.retrieval_query(q, history), q)


class AnswerWithMemoryTests(unittest.TestCase):
    HISTORY = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris.", "meta": {"model": "x"}},
    ]

    def _run(self, decision, history, **kwargs):
        sent = {}

        def fake_post(url, json=None, timeout=None, stream=False):
            sent["url"] = url
            sent["payload"] = json
            return FakeResponse(body={"message": {"role": "assistant", "content": "About 2.1 million."}})

        with mock.patch.object(engine, "get_router", return_value=FakeRouter(decision)), \
             mock.patch.object(engine.requests, "post", side_effect=fake_post):
            result = engine.answer("What is its population?", history=history, **kwargs)
        return result, sent

    def test_history_reaches_ollama_chat(self):
        result, sent = self._run(_decision(), self.HISTORY)
        self.assertTrue(sent["url"].endswith("/api/chat"))
        msgs = sent["payload"]["messages"]
        self.assertEqual([m["content"] for m in msgs],
                         ["What is the capital of France?", "Paris.", "What is its population?"])
        self.assertEqual(result["history_used"], 2)
        self.assertEqual(result["answer"], "About 2.1 million.")
        # UI metadata never leaks into what the model sees.
        self.assertTrue(all(set(m) == {"role", "content"} for m in msgs))

    def test_no_history_is_a_single_turn(self):
        result, sent = self._run(_decision(), None)
        self.assertEqual(len(sent["payload"]["messages"]), 1)
        self.assertEqual(result["history_used"], 0)

    def test_rag_context_rides_on_this_turn_only(self):
        chunks = [{"text": "NEXUS stores vectors in Chroma.",
                   "meta": {"source": "info.md", "chunk_index": 0}, "score": 1.0}]
        with mock.patch.object(engine, "retrieve_chunks", return_value=chunks) as rc:
            result, sent = self._run(_decision(needs_rag=True), self.HISTORY)
        msgs = sent["payload"]["messages"]
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("NEXUS stores vectors in Chroma.", msgs[0]["content"])
        # The context appears once, in the system message, not in history.
        self.assertEqual(sum("Chroma" in m["content"] for m in msgs), 1)
        self.assertEqual(result["sources"][0]["source"], "info.md")
        # The short follow-up was searched together with the earlier question.
        self.assertIn("capital of France", rc.call_args[0][0])

    def test_streaming_reads_chat_chunks(self):
        lines = [json.dumps({"message": {"content": tok}, "done": False}).encode()
                 for tok in ("Par", "is")] + [b"", b"not json",
                                              json.dumps({"done": True}).encode()]
        tokens = []
        with mock.patch.object(engine, "get_router", return_value=FakeRouter(_decision())), \
             mock.patch.object(engine.requests, "post", return_value=FakeResponse(lines=lines)):
            result = engine.answer("capital of France?", on_token=tokens.append)
        self.assertEqual(tokens, ["Par", "is"])
        self.assertEqual(result["answer"], "Paris")


class ChatStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(Path(self.tmp.name) / "t.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip_with_meta(self):
        cid = self.store.create_chat("What is RAG?")
        self.store.add_message(cid, "user", "What is RAG?")
        self.store.add_message(cid, "assistant", "Retrieval...", {"model": "llama3.1:8b",
                                                                 "sources": [{"source": "a.md"}]})
        msgs = self.store.get_messages(cid)
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant"])
        self.assertEqual(msgs[1]["meta"]["sources"][0]["source"], "a.md")
        self.assertNotIn("meta", msgs[0])

    def test_list_most_recent_first_with_counts(self):
        first = self.store.create_chat("first")
        second = self.store.create_chat("second")
        self.store.add_message(first, "user", "bump")  # first is now the latest
        chats = self.store.list_chats()
        self.assertEqual([c["id"] for c in chats], [first, second])
        self.assertEqual(chats[0]["message_count"], 1)
        self.assertEqual(chats[1]["message_count"], 0)

    def test_delete_removes_messages(self):
        cid = self.store.create_chat("x")
        self.store.add_message(cid, "user", "hello")
        self.store.delete_chat(cid)
        self.assertIsNone(self.store.get_chat(cid))
        self.assertEqual(self.store.get_messages(cid), [])

    def test_survives_reopen(self):
        cid = self.store.create_chat("persist me")
        self.store.add_message(cid, "user", "hi")
        reopened = ChatStore(self.store.path)
        self.assertEqual(reopened.get_messages(cid)[0]["content"], "hi")

    def test_titles(self):
        self.assertEqual(make_title(""), "New chat")
        self.assertEqual(make_title("short question\nsecond line"), "short question")
        long = "word " * 40
        title = make_title(long)
        self.assertLessEqual(len(title), 61)
        self.assertTrue(title.endswith("…"))


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_files_give_defaults(self):
        s = config.load_settings(self.dir / "none.toml", self.dir / "none.env")
        self.assertEqual(s.history_messages, 8)
        self.assertEqual(s.history_chars, 8000)
        self.assertEqual(s.db_path, config.PROJECT_DIR / "nexus.db")

    def test_toml_overrides(self):
        toml = self.dir / "n.toml"
        toml.write_text('[memory]\nhistory_messages = 2\n[storage]\ndb_path = "x/chats.db"\n')
        s = config.load_settings(toml, self.dir / "none.env")
        self.assertEqual(s.history_messages, 2)
        self.assertEqual(s.history_chars, 8000)
        self.assertEqual(s.db_path, config.PROJECT_DIR / "x" / "chats.db")

    def test_env_file_parsing_does_not_override_real_env(self):
        env = self.dir / ".env"
        env.write_text(
            "# comment\n\nNEXUS_TEST_A=plain\nexport NEXUS_TEST_B='quoted value'\n"
            'NEXUS_TEST_C="already set"\nnot a pair\n'
        )
        with mock.patch.dict(os.environ, {"NEXUS_TEST_C": "real"}, clear=False):
            found = config.load_env(env)
            self.assertEqual(found["NEXUS_TEST_A"], "plain")
            self.assertEqual(found["NEXUS_TEST_B"], "quoted value")
            self.assertEqual(os.environ["NEXUS_TEST_B"], "quoted value")
            self.assertEqual(os.environ["NEXUS_TEST_C"], "real")
        for key in ("NEXUS_TEST_A", "NEXUS_TEST_B"):
            os.environ.pop(key, None)


if __name__ == "__main__":
    unittest.main()
