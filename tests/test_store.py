import os
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from nexus import store


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

    def test_search_matches_titles_and_message_text(self):
        a = store.create_chat("Embedding models", conn=self.conn)
        b = store.create_chat("Sorting folders", conn=self.conn)
        store.append_message(b, "assistant", "Moved 3 files into Images/", conn=self.conn)

        def found(term):
            return {c["id"] for c in store.list_chats(search=term, conn=self.conn)}

        self.assertEqual(found("embedding"), {a})
        self.assertEqual(found("IMAGES"), {b})
        self.assertEqual(found("   "), {a, b})
        # LIKE wildcards in the query are literal text, not patterns.
        self.assertEqual(found("%"), set())
        self.assertEqual(found("_"), set())

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
