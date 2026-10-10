"""Conversation memory budget and the settings files (nexus.toml, .env).

No Ollama, no network and no embedding model: the router, retriever and HTTP
calls are replaced with fakes, so these run anywhere.
"""

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import isolate  # noqa: F401  (must precede any nexus import)
from fakes import FakeOllama, FakeRetriever, FakeRouter, stream

from nexus import config, engine, ollama, providers, store

_TMP = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
NO_CAPS = {"caps": set(), "context_length": None, "parameter_size": None}


def setUpModule():
    os.environ["NEXUS_DB"] = os.path.join(_TMP.name, "memory-test.db")


def tearDownModule():
    store.close()
    os.environ.pop("NEXUS_DB", None)
    _TMP.cleanup()


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


class DocumentTurnTests(unittest.TestCase):
    """What a cloud model may see of the conversation when documents stay home."""

    def test_grounded_answers_and_their_questions_are_dropped(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi", "meta": {"needs_rag": False}},
            {"role": "user", "content": "what do my notes say?"},
            {"role": "assistant", "content": "They say X.", "meta": {"sources": [{"source": "n.md"}]}},
            {"role": "user", "content": "thanks"},
        ]
        kept = engine.without_document_turns(history)
        self.assertEqual([m["content"] for m in kept], ["hello", "hi", "thanks"])

    def test_plain_history_is_untouched(self):
        history = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        self.assertEqual(engine.without_document_turns(history), history)
        self.assertEqual(engine.without_document_turns(None), [])


class AnswerWithMemoryTests(unittest.TestCase):
    HISTORY = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris.", "meta": {"model": "x"}, "id": 7},
    ]

    def run_answer(self, history, settings=None, **kwargs):
        sent = FakeOllama({"a": stream("About 2.1 million.")})
        patches = [
            mock.patch.object(engine, "get_router", lambda: FakeRouter()),
            mock.patch.object(engine, "get_retriever", lambda: FakeRetriever(())),
            mock.patch.object(ollama.requests, "post", sent),
            mock.patch.object(providers, "capabilities", lambda m: NO_CAPS),
        ]
        if settings is not None:
            patches.append(mock.patch.object(engine, "get_settings", return_value=settings))
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        result = engine.answer("What is its population?", history=history, **kwargs)
        return result, sent.calls[-1]["json"]["messages"]

    def test_history_reaches_the_model_without_ui_metadata(self):
        result, msgs = self.run_answer(self.HISTORY)
        self.assertEqual([m["content"] for m in msgs],
                         ["What is the capital of France?", "Paris.", "What is its population?"])
        self.assertEqual(result["history_used"], 2)
        self.assertTrue(all(set(m) == {"role", "content"} for m in msgs))

    def test_no_history_is_a_single_turn(self):
        result, msgs = self.run_answer(None)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(result["history_used"], 0)

    def test_memory_budget_from_settings_caps_what_is_sent(self):
        long_chat = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
                     for i in range(12)]
        settings = replace(config.Settings(), history_messages=2)
        result, msgs = self.run_answer(long_chat, settings=settings)
        self.assertEqual([m["content"] for m in msgs], ["m10", "m11", "What is its population?"])
        self.assertEqual(result["history_used"], 2)

    def test_zero_budget_sends_only_the_question(self):
        settings = replace(config.Settings(), history_messages=0)
        _result, msgs = self.run_answer(self.HISTORY, settings=settings)
        self.assertEqual(len(msgs), 1)


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
        self.assertIsNone(s.db_path)
        self.assertEqual(s.cloud_mode, "off")
        self.assertFalse(s.allow_docs_to_cloud)
        self.assertFalse(s.allow_paid)

    def test_toml_overrides(self):
        toml = self.dir / "n.toml"
        toml.write_text('[memory]\nhistory_messages = 2\n[storage]\ndb_path = "x/chats.db"\n')
        s = config.load_settings(toml, self.dir / "none.env")
        self.assertEqual(s.history_messages, 2)
        self.assertEqual(s.history_chars, 8000)
        self.assertEqual(s.db_path, config.PROJECT_DIR / "x" / "chats.db")

    def test_database_path_order_is_env_then_toml_then_data_folder(self):
        settings = replace(config.Settings(), db_path=self.dir / "from-toml.db")
        with mock.patch.object(config, "get_settings", return_value=settings):
            with mock.patch.dict(os.environ, {"NEXUS_DB": str(self.dir / "from-env.db")}):
                self.assertEqual(config.db_path().name, "from-env.db")
            with mock.patch.dict(os.environ, {"NEXUS_DB": ""}):
                self.assertEqual(config.db_path().name, "from-toml.db")
        with mock.patch.object(config, "get_settings", return_value=config.Settings()), \
                mock.patch.dict(os.environ, {"NEXUS_DB": ""}):
            self.assertEqual(config.db_path(), config.DATA_DIR / "nexus.db")

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

    def test_a_bad_cloud_mode_falls_back_to_off(self):
        toml = self.dir / "n.toml"
        toml.write_text('[cloud]\nmode = "everything"\n')
        self.assertEqual(config.load_settings(toml, self.dir / "none.env").cloud_mode, "off")


if __name__ == "__main__":
    unittest.main()
