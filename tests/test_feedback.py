"""Feedback, Arena and the leaderboard (Phase 3).

Runs against demos/mock_ollama.py in-process, including real HTTP calls to
the server's endpoints, so no models or network are needed.
"""

import json
import random
import sys
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demos"))

import fakes  # noqa: E402
import isolate  # noqa: F401  (must precede any nexus import)
import mock_ollama  # noqa: E402

from nexus import (
    config,  # noqa: E402
    engine,  # noqa: E402
    feedback,  # noqa: E402
    providers,  # noqa: E402
    )

LOCAL = ["llama3.1:8b", "deepseek-r1:7b", "qwen2.5-coder:7b"]


class FakeRouter:
    """Real planning over a fixed set of installed models."""

    def __init__(self, task="general", installed=LOCAL, auto_task=None):
        self.task, self.installed, self.auto_task = task, installed, auto_task

    def route(self, question, force_task=None, policy=None, needs_image=False,
              rag_override=None):
        task = force_task or self.task
        if task == "system_agent":
            chain = [{"model": "pc-toolkit", "provider": "toolkit", "local": True,
                      "score": 1.0, "reason": "toolkit"}]
        else:
            avail = {"ollama": self.installed, "loaded": [], "cloud": {}}
            chain = [{"model": c.model, "provider": c.spec.provider, "local": c.spec.is_local,
                      "score": c.score, "reason": c.reason}
                     for c in providers.plan(task, question, 0.2, avail, policy=policy)]
        return {"task": task, "auto_task": self.auto_task or self.task,
                "model": chain[0]["model"] if chain else None,
                "provider": chain[0]["provider"] if chain else None, "chain": chain,
                "complexity": 0.2, "available": {"ollama": self.installed},
                "needs_rag": bool(rag_override), "confidence": 1.0,
                "scores": {task: 1.0}, "reason": "test"}

    def _needs_rag(self, question):
        return "document" in question.lower()


class FeedbackTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ollama, cls.ollama_url = mock_ollama.start_in_thread()

    @classmethod
    def tearDownClass(cls):
        cls.ollama.shutdown()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = fakes.use_db(Path(self.tmp.name) / "t.db")
        self.patches = [
            mock.patch.object(config, "OLLAMA_URL", self.ollama_url),
            mock.patch.object(engine, "get_router", return_value=FakeRouter()),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        fakes.restore_db()
        self.tmp.cleanup()

    def answer_message(self, model="llama3.1:8b", task="general"):
        cid = self.store.create_chat("Explain a hash table")
        self.store.add_message(cid, "user", "Explain a hash table")
        mid = self.store.add_message(cid, "assistant", "It maps keys to slots.",
                                     {"model": model, "provider": "ollama", "task": task,
                                      "complexity": 0.2})
        return cid, mid


class RatingTests(FeedbackTestCase):
    def test_rating_is_stored_with_its_question(self):
        _, mid = self.answer_message()
        feedback.rate(mid, 1)
        (row,) = feedback.ratings()
        self.assertEqual((row["rating"], row["question"], row["model"], row["task"]),
                         (1, "Explain a hash table", "llama3.1:8b", "general"))
        self.assertEqual(self.store.get_message(mid)["meta"]["rating"], 1)

    def test_changing_your_mind_overwrites_and_zero_removes(self):
        _, mid = self.answer_message()
        feedback.rate(mid, 1)
        feedback.rate(mid, -1, "too long")
        (row,) = feedback.ratings()
        self.assertEqual((row["rating"], row["reason"]), (-1, "too long"))
        feedback.rate(mid, 0)
        self.assertEqual(feedback.ratings(), [])
        self.assertIsNone(self.store.get_message(mid)["meta"]["rating"])

    def test_reason_only_kept_for_thumbs_down(self):
        _, mid = self.answer_message()
        feedback.rate(mid, 1, "wrong")
        self.assertIsNone(feedback.ratings()[0]["reason"])

    def test_unknown_reason_becomes_other_and_bad_input_is_rejected(self):
        _, mid = self.answer_message()
        feedback.rate(mid, -1, "made me sad")
        self.assertEqual(feedback.ratings()[0]["reason"], "other")
        with self.assertRaises(ValueError):
            feedback.rate(mid, 5)
        with self.assertRaises(ValueError):
            feedback.rate(999, 1)


class SignalTests(FeedbackTestCase):
    def test_task_override_is_recorded(self):
        with mock.patch.object(engine, "get_router",
                               return_value=FakeRouter(task="general", auto_task="general")):
            engine.answer("Explain a hash table", options=engine.Options(force_task="coding"))
        (sig,) = feedback.signals("task_override")
        self.assertEqual((sig["task"], sig["value"]), ("general", "coding"))

    def test_rag_override_recorded_only_when_it_disagrees(self):
        engine.answer("Explain a hash table", options=engine.Options(rag_mode="never"))
        self.assertEqual(feedback.signals("rag_override"), [])  # agreed: no docs
        with mock.patch.object(engine, "get_retriever", return_value=fakes.FakeRetriever([])):
            engine.answer("Explain a hash table", options=engine.Options(rag_mode="always"))
        (sig,) = feedback.signals("rag_override")
        self.assertEqual(sig["value"], "always")

    def test_a_correction_from_the_ui_is_a_task_override(self):
        feedback.correct_task("  why is my loop slow?  ", "reasoning", "coding", message_id=7)
        [signal] = feedback.signals("task_override")
        self.assertEqual((signal["question"], signal["task"], signal["value"], signal["message_id"]),
                         ("why is my loop slow?", "reasoning", "coding", 7))
        with self.assertRaises(ValueError):
            feedback.correct_task("q", "coding", "coding")   # not a correction
        with self.assertRaises(ValueError):
            feedback.correct_task("   ", "general", "coding")

    def test_no_override_no_signal(self):
        engine.answer("Explain a hash table")
        self.assertEqual(feedback.signals(), [])

    def test_unknown_signal_kind_rejected(self):
        with self.assertRaises(ValueError):
            feedback.record_signal("vibes")


class PairTests(unittest.TestCase):
    CHAIN = [
        {"model": "llama3.1:8b", "provider": "ollama", "local": True},
        {"model": "deepseek-r1:7b", "provider": "ollama", "local": True},
        {"model": "llama3.1:8b", "provider": "ollama", "local": True},  # duplicate
        {"model": "gemini-3.7-flash", "provider": "gemini", "local": False},
    ]

    def test_pair_is_two_models_including_first_choice(self):
        for seed in range(20):
            a, b = feedback.pick_pair(self.CHAIN[:2], random.Random(seed))
            self.assertNotEqual(a["model"], b["model"])
            self.assertIn("llama3.1:8b", (a["model"], b["model"]))

    def test_challenger_prefers_the_other_side(self):
        for seed in range(20):
            pair = feedback.pick_pair(self.CHAIN, random.Random(seed))
            self.assertEqual({p["model"] for p in pair}, {"llama3.1:8b", "gemini-3.7-flash"})

    def test_first_choice_appears_on_both_sides(self):
        positions = {feedback.pick_pair(self.CHAIN[:2], random.Random(s))[0]["model"]
                     for s in range(30)}
        self.assertEqual(positions, {"llama3.1:8b", "deepseek-r1:7b"})

    def test_not_enough_models(self):
        self.assertIsNone(feedback.pick_pair(self.CHAIN[:1]))
        self.assertIsNone(feedback.pick_pair([{"model": "pc-toolkit", "provider": "toolkit"},
                                              {"model": "x", "provider": "ollama"}]))


class EloTests(unittest.TestCase):
    def battle(self, a, b, winner):
        return {"model_a": a, "model_b": b, "winner": winner}

    def test_win_between_equals_moves_sixteen_points(self):
        r = feedback.elo([self.battle("x", "y", "a")])
        self.assertAlmostEqual(r["x"], 1016.0)
        self.assertAlmostEqual(r["y"], 984.0)

    def test_tie_between_equals_and_both_bad_change_nothing(self):
        r = feedback.elo([self.battle("x", "y", "tie"), self.battle("x", "y", "both_bad")])
        self.assertEqual((r["x"], r["y"]), (1000.0, 1000.0))

    def test_upset_gains_more_than_expected_win(self):
        history = [self.battle("strong", "weak", "a")] * 10
        before = feedback.elo(history)
        favourite = feedback.elo(history + [self.battle("strong", "weak", "a")])
        upset = feedback.elo(history + [self.battle("strong", "weak", "b")])
        gain_expected = favourite["strong"] - before["strong"]
        gain_upset = upset["weak"] - before["weak"]
        self.assertGreater(gain_upset, gain_expected)

    def test_points_are_conserved(self):
        history = [self.battle("x", "y", "a"), self.battle("y", "z", "tie"),
                   self.battle("z", "x", "b")]
        self.assertAlmostEqual(sum(feedback.elo(history).values()), 3000.0)


class ArenaTests(FeedbackTestCase):
    def test_arena_answers_with_two_models_and_saves_the_battle(self):
        out = engine.arena("Explain a hash table", rng=random.Random(1))
        self.assertNotEqual(out["a"]["model"], out["b"]["model"])
        self.assertNotEqual(out["a"]["answer"], out["b"]["answer"])
        battle = feedback.get_battle(out["battle_id"])
        self.assertEqual(battle["prompt"], "Explain a hash table")
        self.assertIsNone(battle["winner"])
        self.assertEqual(feedback.battles(), [])  # undecided battles don't count yet

    def test_vote_once_and_leaderboard(self):
        out = engine.arena("Explain a hash table", rng=random.Random(1))
        feedback.vote(out["battle_id"], "a")
        with self.assertRaises(ValueError):
            feedback.vote(out["battle_id"], "b")
        with self.assertRaises(ValueError):
            feedback.vote(out["battle_id"], "maybe")
        board = feedback.leaderboard()
        self.assertEqual(board[0]["model"], out["a"]["model"])
        self.assertEqual((board[0]["wins"], board[1]["losses"]), (1, 1))
        self.assertEqual(feedback.leaderboard("coding"), [])
        self.assertFalse(board[0]["settled"])  # one vote is not a verdict

    def test_leaderboard_includes_thumbs(self):
        _, mid = self.answer_message(model="qwen2.5-coder:7b")
        feedback.rate(mid, 1)
        (row,) = feedback.leaderboard()
        self.assertEqual((row["model"], row["thumbs_up"], row["approval"], row["battles"]),
                         ("qwen2.5-coder:7b", 1, 1.0, 0))

    def test_pc_actions_and_single_model_are_refused(self):
        with mock.patch.object(engine, "get_router", return_value=FakeRouter("system_agent")):
            self.assertIn("error", engine.arena("sort my downloads"))
        with mock.patch.object(engine, "get_router",
                               return_value=FakeRouter(installed=["llama3.1:8b"])):
            self.assertIn("at least two", engine.arena("Explain a hash table")["error"])


class ServerTests(FeedbackTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from nexus import server

        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        super().tearDownClass()

    def call(self, path, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(self.base + path, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as res:
            return json.loads(res.read())

    def test_blind_battle_then_reveal_and_chat_continues(self):
        out = self.call("/api/arena", {"question": "Explain a hash table"})
        # Names stay hidden until the vote: no model keys, no model names.
        self.assertEqual(set(out["a"]) | set(out["b"]), {"answer"})
        for name in LOCAL:
            self.assertNotIn(name, json.dumps(out))
        voted = self.call("/api/arena/vote", {"battle_id": out["battle_id"], "winner": "b"})
        self.assertNotEqual(voted["a"]["model"], voted["b"]["model"])
        msgs = self.store.get_messages(out["chat_id"])
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant"])
        self.assertEqual(msgs[1]["content"], out["b"]["answer"])
        self.assertEqual(msgs[1]["meta"]["model"], voted["b"]["model"])
        board = self.call("/api/leaderboard")
        self.assertEqual(board["summary"]["battles"], 1)
        self.assertEqual(board["rows"][0]["model"], voted["b"]["model"])

    def test_feedback_endpoint(self):
        reply = self.call("/api/chat", {"question": "Explain a hash table"})
        self.call("/api/feedback", {"message_id": reply["message_id"], "rating": -1,
                                    "reason": "too long"})
        self.assertEqual(feedback.ratings()[0]["reason"], "too long")


if __name__ == "__main__":
    unittest.main()
