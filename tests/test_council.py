"""Model council (Phase 5).

Runs against the in-process mock Ollama and mock cloud (whose mock judge
returns the JSON verdict a real judge would), including real HTTP calls to
the server.
"""

import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.request
from dataclasses import replace
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demos"))

import cloud  # noqa: E402
import config  # noqa: E402
import council  # noqa: E402
import engine  # noqa: E402
import feedback  # noqa: E402
import providers  # noqa: E402
import store  # noqa: E402
import mock_cloud  # noqa: E402
import mock_ollama  # noqa: E402

LOCAL = ["llama3.1:8b", "deepseek-r1:7b", "qwen2.5-coder:7b"]


class FakeRouter:
    def __init__(self, task="general", needs_rag=False, complexity=0.2, chain=None,
                 auto_task=None):
        self.task, self.needs_rag, self.complexity = task, needs_rag, complexity
        self.chain, self.auto_task = chain, auto_task

    def route(self, question, force_task=None, policy=None, needs_image=False, rag_override=None):
        task = force_task or self.task
        needs_rag = self.needs_rag if rag_override is None else rag_override
        if task == "system_agent":
            chain = [{"model": "pc-toolkit", "provider": "toolkit", "local": True}]
        elif self.chain is not None:
            chain = self.chain
        else:
            avail = providers.availability(LOCAL, [])
            chain = [{"model": c.model, "provider": c.spec.provider, "local": c.spec.is_local,
                      "score": c.score, "reason": c.reason}
                     for c in providers.plan(task, question, needs_rag, self.complexity, avail,
                                             policy=policy)]
        return {"task": task, "auto_task": self.auto_task or task, "chain": chain,
                "model": chain[0]["model"] if chain else None,
                "provider": chain[0]["provider"] if chain else None,
                "complexity": self.complexity, "available": {"ollama": LOCAL},
                "needs_rag": needs_rag, "confidence": 1.0, "scores": {task: 1.0},
                "reason": "test", "p_strong": None}

    def _needs_rag(self, q):
        return False


class CouncilTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ollama, cls.ollama_url = mock_ollama.start_in_thread()
        cls.cloud, cls.cloud_url = mock_cloud.start_in_thread()

    @classmethod
    def tearDownClass(cls):
        cls.ollama.shutdown()
        cls.cloud.shutdown()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = store.ChatStore(Path(self.tmp.name) / "t.db")
        self._saved = store._default
        store._default = self.store
        mock_cloud.reset()
        cloud.reset_state()
        self.settings = config.Settings(
            daily_limits=dict(config.DEFAULT_DAILY_LIMITS),
            provider_overrides={p: {"base_url": f"{self.cloud_url}/{p}/v1"}
                                for p in cloud.PROVIDERS})
        self.patches = [
            mock.patch.object(engine, "OLLAMA_CHAT_URL", f"{self.ollama_url}/api/chat"),
            mock.patch.object(engine, "get_router", return_value=FakeRouter()),
            mock.patch.dict(os.environ, {"GEMINI_API_KEY": "k", "GROQ_API_KEY": "k"}),
        ] + [mock.patch.object(m, "get_settings", side_effect=lambda: self.settings)
             for m in (cloud, engine)]
        for p in self.patches:
            p.start()
        cloud.cloud_specs.cache_clear()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        store._default = self._saved
        cloud.reset_state()
        cloud.cloud_specs.cache_clear()
        self.tmp.cleanup()


class PureTests(unittest.TestCase):
    def test_pick_members(self):
        chain = [{"model": "a", "provider": "ollama"}, {"model": "a", "provider": "ollama"},
                 {"model": "pc-toolkit", "provider": "toolkit"}, {"model": "b", "provider": "ollama"},
                 {"model": "c", "provider": "gemini"}, {"model": "d", "provider": "ollama"}]
        self.assertEqual([m["model"] for m in council.pick_members(chain, 3)], ["a", "b", "c"])

    def test_parse_verdict(self):
        text = ('Here you go: {"agreements": ["x", " ", "y"], "disagreements": ['
                '{"point": "speed", "positions": {"Answer 1": "fast", "2": "slow", "9": "?"}},'
                '{"positions": {"1": "no point"}}], "final": "merged"} thanks')
        v = council.parse_verdict(text, 2)
        self.assertEqual(v["agreements"], ["x", "y"])
        self.assertEqual(v["disagreements"], [{"point": "speed", "positions": {1: "fast", 2: "slow"}}])
        self.assertEqual(v["final"], "merged")
        self.assertIsNone(council.parse_verdict('{"agreements": []}', 2))
        self.assertIsNone(council.parse_verdict("not json", 2))

    def test_agreement_score(self):
        same = council.agreement(["hash tables map keys to values"] * 3)
        self.assertAlmostEqual(same["score"], 1.0, places=3)
        apart = council.agreement(["hash tables map keys", "photosynthesis needs sunlight"])
        self.assertLess(apart["score"], 0.2)
        mixed = council.agreement(["keys map to values quickly", "keys map to values",
                                   "the weather is sunny today"])
        self.assertIn(mixed["central"], (0, 1))
        self.assertIsNone(council.agreement(["only one"])["score"])


class EngineCouncilTests(CouncilTestCase):
    def test_three_local_members_and_a_judge(self):
        out = engine.council("Explain a hash table")
        self.assertEqual(len(out["used"]), 3)
        self.assertEqual(len(set(out["used"])), 3)
        self.assertTrue(out["judged"])
        self.assertIn(out["judge"], LOCAL)
        self.assertIn("mock judge", out["answer"])
        self.assertTrue(out["agreements"])
        self.assertEqual(out["disagreements"][0]["point"], "How fast lookups are")
        meta = engine.council_message_meta(out)
        self.assertTrue(meta["local"])
        self.assertEqual(meta["council"]["used"], out["used"])

    def test_judge_failure_falls_back_to_most_central_answer(self):
        with mock.patch.object(council, "parse_verdict", return_value=None):
            out = engine.council("Explain a hash table")
        self.assertFalse(out["judged"])
        self.assertIn("No judge could compare", out["note"])
        self.assertIn(out["answer"], [m["answer"] for m in out["members"]])

    def test_cloud_members_run_in_parallel_local_ones_in_order(self):
        chain = [{"model": "llama3.1:8b", "provider": "ollama", "local": True},
                 {"model": "gemini-3.7-flash", "provider": "gemini", "local": False},
                 {"model": "openai/gpt-oss-20b", "provider": "groq", "local": False}]
        threads = {}
        real = engine.answer

        def spy(question, options=None, **kw):
            threads[options.force_model] = threading.current_thread() is threading.main_thread()
            return real(question, options=options, **kw)

        with mock.patch.object(engine, "get_router", return_value=FakeRouter(chain=chain)), \
             mock.patch.object(engine, "answer", side_effect=spy):
            out = engine.council("Explain a hash table",
                                 engine.Options(cloud_mode="allowed"))
        self.assertEqual(threads, {"llama3.1:8b": True, "gemini-3.7-flash": False,
                                   "openai/gpt-oss-20b": False})
        self.assertEqual(len(out["used"]), 3)
        self.assertFalse(engine.council_message_meta(out)["local"])

    def test_document_questions_never_reach_cloud_members_or_judge(self):
        chain = [{"model": "gemini-3.7-flash", "provider": "gemini", "local": False},
                 {"model": "llama3.1:8b", "provider": "ollama", "local": True},
                 {"model": "deepseek-r1:7b", "provider": "ollama", "local": True}]
        with mock.patch.object(engine, "get_router",
                               return_value=FakeRouter(chain=chain, needs_rag=True)), \
             mock.patch.object(engine, "retrieve_chunks", return_value=[]):
            out = engine.council("What do my notes say about hashing?",
                                 engine.Options(cloud_mode="allowed"))
        self.assertEqual(mock_cloud.STATE["requests"], [])
        self.assertTrue(all(providers.spec_by_name(m).is_local for m in out["used"]))
        self.assertIn(out["judge"], LOCAL)

    def test_one_correction_recorded_not_one_per_member(self):
        with mock.patch.object(engine, "get_router",
                               return_value=FakeRouter(auto_task="general")):
            engine.council("Explain a hash table", engine.Options(force_task="reasoning"))
        self.assertEqual(len(feedback.signals("task_override")), 1)

    def test_refusals(self):
        with mock.patch.object(engine, "get_router", return_value=FakeRouter("system_agent")):
            self.assertIn("error", engine.council("sort my downloads"))
        one = [{"model": "llama3.1:8b", "provider": "ollama", "local": True}]
        with mock.patch.object(engine, "get_router", return_value=FakeRouter(chain=one)):
            self.assertIn("at least two", engine.council("hi")["error"])

    def test_auto_convene_only_when_switched_on_and_hard(self):
        hard = FakeRouter(complexity=0.9)
        with mock.patch.object(engine, "get_router", return_value=hard):
            self.assertFalse(engine.council_recommended("hard question"))
            self.settings = replace(self.settings, council_auto="hard")
            self.assertTrue(engine.council_recommended("hard question"))
        with mock.patch.object(engine, "get_router", return_value=FakeRouter(complexity=0.1)):
            self.assertFalse(engine.council_recommended("easy"))


class ServerCouncilTests(CouncilTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        import server

        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        super().tearDownClass()

    def call(self, path, payload):
        req = urllib.request.Request(self.base + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as res:
            return json.loads(res.read())

    def test_council_answer_joins_the_chat(self):
        out = self.call("/api/council", {"question": "Explain a hash table"})
        msgs = self.store.get_messages(out["chat_id"])
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant"])
        self.assertEqual(msgs[1]["meta"]["model"], "council")
        self.assertEqual(len(msgs[1]["meta"]["council"]["members"]), 3)
        feedback.rate(out["message_id"], 1)  # councils can be rated like any answer
        self.assertEqual(feedback.ratings()[0]["model"], "council")

    def test_chat_convenes_council_by_itself_when_auto_hard(self):
        self.settings = replace(self.settings, council_auto="hard")
        with mock.patch.object(engine, "get_router", return_value=FakeRouter(complexity=0.9)):
            out = self.call("/api/chat", {"question": "Explain a hash table"})
        self.assertEqual(out["model"], "council")
        self.assertTrue(out["council"]["auto"])


if __name__ == "__main__":
    unittest.main()
