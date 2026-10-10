import os
import tempfile
import time
import unittest
from pathlib import Path

import isolate  # noqa: F401  (must precede any nexus import)
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


class MessageIdTests(StoreTestCase):
    """Ratings and later truth checks attach to a message by its id."""

    def test_append_returns_the_id_load_messages_reports(self):
        chat = store.create_chat("ids", conn=self.conn)
        first = store.append_message(chat, "user", "q", conn=self.conn)
        second = store.append_message(chat, "assistant", "a", {"model": "m"}, conn=self.conn)
        self.assertEqual([m["id"] for m in store.load_messages(chat, conn=self.conn)],
                         [first, second])
        message = store.get_message(second, conn=self.conn)
        self.assertEqual((message["chat_id"], message["content"]), (chat, "a"))
        self.assertIsNone(store.get_message(9999, conn=self.conn))

    def test_update_meta_merges(self):
        chat = store.create_chat("meta", conn=self.conn)
        mid = store.append_message(chat, "assistant", "a", {"model": "m"}, conn=self.conn)
        store.update_meta(mid, {"truth": {"supported": 2}}, conn=self.conn)
        store.update_meta(9999, {"ignored": True}, conn=self.conn)  # no such message: no error
        self.assertEqual(store.get_message(mid, conn=self.conn)["meta"],
                         {"model": "m", "truth": {"supported": 2}})


class UsageTests(StoreTestCase):
    def test_requests_are_counted_per_provider_per_day(self):
        store.record_usage("groq", 100, 40, day="2026-10-01", conn=self.conn)
        store.record_usage("groq", 10, 5, day="2026-10-01", conn=self.conn)
        store.record_usage("groq", 1, 1, day="2026-10-02", conn=self.conn)
        self.assertEqual(store.usage_today("groq", day="2026-10-01", conn=self.conn), 2)
        self.assertEqual(store.usage_today("gemini", day="2026-10-01", conn=self.conn), 0)
        self.assertEqual(store.usage_for_day("2026-10-01", conn=self.conn),
                         {"groq": {"requests": 2, "chars_in": 110, "chars_out": 45}})


class HandleAndUpgradeTests(unittest.TestCase):
    def test_chat_store_handle_shares_the_file_with_module_functions(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            handle = store.ChatStore(Path(d) / "h.db")
            chat = handle.create_chat("via handle")
            mid = handle.add_message(chat, "assistant", "a", {"k": 1})
            handle.update_meta(mid, {"j": 2})
            self.assertEqual(handle.get_message(mid)["meta"], {"k": 1, "j": 2})
            self.assertEqual([m["id"] for m in handle.get_messages(chat)], [mid])
            handle.record_usage("groq", 5, 5)
            self.assertEqual(handle.usage_today("groq"), 1)
            conn = store.connect(handle.path)
            try:
                self.assertEqual(store.list_chats(conn=conn)[0]["title"], "via handle")
            finally:
                conn.close()

    def test_a_version_1_database_gains_the_new_tables_and_keeps_its_chats(self):
        import sqlite3

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            path = Path(d) / "old.db"
            old = sqlite3.connect(path)
            old.executescript(
                "CREATE TABLE chats (id TEXT PRIMARY KEY, title TEXT NOT NULL, "
                "created REAL NOT NULL, updated REAL NOT NULL);"
                "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id TEXT NOT NULL, "
                "role TEXT NOT NULL, content TEXT NOT NULL, meta TEXT, created REAL NOT NULL);"
                "INSERT INTO chats VALUES ('c1', 'kept', 1.0, 1.0);"
                "PRAGMA user_version = 1;"
            )
            old.commit()
            old.close()
            conn = store.connect(path)
            try:
                tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                self.assertTrue({"usage", "feedback", "battles", "inbox", "cards"} <= tables)
                self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], store.SCHEMA_VERSION)
                self.assertEqual(store.list_chats(conn=conn)[0]["title"], "kept")
            finally:
                conn.close()


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
