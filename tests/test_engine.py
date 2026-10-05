import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_TMP = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
os.environ["NEXUS_DB"] = os.path.join(_TMP.name, "engine-test.db")

from fakes import FakeOllama, FakeRetriever, FakeRouter, chunk, stream  # noqa: E402

from nexus import (  # noqa: E402
    engine,
    ollama,
    providers,  # noqa: E402
    store,  # noqa: E402
)

MESSAGES = [{"role": "user", "content": "hi"}]


class GuardTests(unittest.TestCase):
    def test_guard_wraps_callback_errors(self):
        def boom(_):
            raise ValueError("ui gone")

        with self.assertRaises(engine._CallbackRaised) as caught:
            engine._guard(boom)("x")
        self.assertIsInstance(caught.exception.original, ValueError)
        self.assertIsNone(engine._guard(None))


NO_CAPS = {"caps": set(), "context_length": None, "parameter_size": None}


def tearDownModule():
    store.close()
    _TMP.cleanup()


class AnswerTests(unittest.TestCase):
    def use(self, *, chain=("a",), task="general", needs_rag=False, scripts=None,
            caps=None, chunks=()):
        self.router = FakeRouter(task=task, chain=chain, needs_rag=needs_rag)
        self.retriever = FakeRetriever(chunks)
        self.ollama = FakeOllama(scripts or {m: stream("ok") for m in chain})
        caps = caps or {}
        for patcher in (
            mock.patch.object(engine, "get_router", lambda: self.router),
            mock.patch.object(engine, "get_retriever", lambda: self.retriever),
            mock.patch.object(ollama.requests, "post", self.ollama),
            mock.patch.object(providers, "capabilities",
                              lambda m: {**NO_CAPS, **caps.get(m, {})}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def sent(self, call=-1):
        return self.ollama.calls[call]["json"]

    def test_falls_through_to_the_next_model(self):
        self.use(chain=("a", "b"), scripts={"a": ConnectionError("down"), "b": stream("ok")})
        result = engine.answer("hello")
        self.assertEqual(result["model"], "b")
        self.assertEqual([a["model"] for a in result["attempts"]], ["a", "b"])
        self.assertIn("Auto-switched", result["info"])

    def test_pinned_model_goes_first(self):
        self.use(chain=("a", "b"))
        engine.answer("hello", options=engine.Options(force_model="b"))
        self.assertEqual(self.sent(0)["model"], "b")

    def test_rag_never_skips_retrieval(self):
        self.use(needs_rag=True, chunks=[chunk(0)])
        result = engine.answer("what does the readme say",
                               options=engine.Options(rag_mode="never"))
        self.assertEqual(self.retriever.queries, [])
        self.assertEqual(result["sources"], [])

    def test_rag_always_grounds_the_newest_message(self):
        self.use(needs_rag=False, chunks=[chunk(0)])
        result = engine.answer("hello", options=engine.Options(rag_mode="always"))
        self.assertEqual(self.retriever.queries, ["hello"])
        self.assertEqual(len(result["sources"]), 1)
        self.assertIn("chunk0", self.sent()["messages"][-1]["content"])

    def test_history_goes_before_the_question_in_order(self):
        self.use()
        history = [
            {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"},
        ]
        engine.answer("q3", history=history)
        self.assertEqual(self.sent()["messages"],
                         history + [{"role": "user", "content": "q3"}])

    def test_history_is_trimmed_newest_first_to_fit(self):
        # context 1400 -> budget 376 tokens; each message costs 100 + 4.
        self.use(caps={"a": {"context_length": 1400}})
        history = [{"role": "user" if i % 2 == 0 else "assistant",
                    "content": f"m{i} " + "x" * 297} for i in range(10)]
        engine.answer("q?", history=history)
        self.assertEqual(self.sent()["messages"],
                         history[-3:] + [{"role": "user", "content": "q?"}])
        self.assertEqual(self.sent()["options"]["num_ctx"], 1400)

    def test_chunks_that_do_not_fit_are_dropped_and_counted(self):
        self.use(needs_rag=True, caps={"a": {"context_length": 2048}},
                 chunks=[chunk(i) for i in range(20)])
        result = engine.answer("what does the readme say",
                               options=engine.Options(top_k=20))
        self.assertGreater(result["chunks_dropped"], 0)
        self.assertEqual(len(result["sources"]) + result["chunks_dropped"], 20)
        self.assertLessEqual(
            engine.estimate_tokens(self.sent()["messages"][-1]["content"]),
            2048 - engine.REPLY_RESERVE,
        )
        self.assertIn("left out", result["info"])

    def test_truncation_is_flagged_when_the_window_filled(self):
        self.use(caps={"a": {"context_length": 2048}},
                 scripts={"a": stream("ok", done_extra={"prompt_eval_count": 2030})})
        result = engine.answer("hello")
        self.assertTrue(result["truncated"])
        self.assertIn("context window", result["info"])

    def test_no_truncation_flag_normally(self):
        self.use()
        self.assertFalse(engine.answer("hello")["truncated"])

    def test_think_is_sent_only_to_thinking_models(self):
        self.use(chain=("a", "b"),
                 scripts={"a": ConnectionError("down"), "b": stream("ok")},
                 caps={"a": {"caps": {"completion", "thinking"}}})
        engine.answer("hello")
        self.assertIs(self.sent(0).get("think"), True)
        self.assertNotIn("think", self.sent(1))

    def test_thinking_reaches_the_result(self):
        self.use(scripts={"a": stream("ok", thinking=("hmm",))})
        self.assertEqual(engine.answer("hello")["thinking"], "hmm")

    def test_callback_errors_are_not_model_failures(self):
        self.use(chain=("a", "b"))

        class UiGone(Exception):
            pass

        def on_token(_):
            raise UiGone()

        with self.assertRaises(UiGone):
            engine.answer("hello", on_token=on_token)
        self.assertEqual(len(self.ollama.calls), 1)  # b was never tried

    def test_follow_up_borrows_the_previous_question_for_search_only(self):
        self.use(chunks=[chunk(0)])
        history = [{"role": "user", "content": "Which embedding model does NEXUS use?"},
                   {"role": "assistant", "content": "all-MiniLM-L6-v2."}]
        engine.answer("and its dimension?", history=history,
                      options=engine.Options(rag_mode="always"))
        self.assertEqual(self.retriever.queries,
                         ["Which embedding model does NEXUS use? and its dimension?"])
        self.assertIn("User question: and its dimension?",
                      self.sent()["messages"][-1]["content"])

    def test_one_turn_is_recorded_per_answer(self):
        self.use()
        engine.answer("hello", chat_id="chat-1")
        turn = store.recent_turns(1)[0]
        self.assertEqual((turn["model"], turn["chat_id"]), ("a", "chat-1"))
        self.assertEqual(turn["tokens_per_s"], 20.0)
        self.assertIsNotNone(turn["total_ms"])

    def test_a_turn_is_recorded_even_when_every_model_fails(self):
        self.use(scripts={"a": ConnectionError("down")})
        with self.assertRaises(RuntimeError):
            engine.answer("hello")
        turn = store.recent_turns(1)[0]
        self.assertIsNone(turn["model"])
        self.assertTrue(turn["error"].startswith("Every available model failed"))

    # -- relevance probe ------------------------------------------------------
    def relevant(self, score):
        return dict(chunk(0), rerank_score=score, score=score)

    def test_relevant_documents_ground_a_question_that_never_named_them(self):
        # "which model writes the answer?" has no "document"/"project" keyword,
        # so the keyword gate alone answered it from general knowledge.
        self.use(needs_rag=False, chunks=[self.relevant(2.5)])
        result = engine.answer("which model writes the final answer?")
        self.assertTrue(result["needs_rag"])
        self.assertEqual(len(result["sources"]), 1)
        self.assertIn("looked relevant", result["rag_reason"])
        self.assertIn("chunk0", self.sent()["messages"][-1]["content"])

    def test_irrelevant_documents_stay_out_of_the_prompt(self):
        self.use(needs_rag=False, chunks=[self.relevant(-9.0)])
        result = engine.answer("what is the capital of France?")
        self.assertFalse(result["needs_rag"])
        self.assertEqual(result["sources"], [])
        self.assertEqual(self.sent()["messages"][-1]["content"], "what is the capital of France?")
        self.assertIsNone(result["info"])  # a probe that finds nothing is silent

    def test_no_probe_without_reranking(self):
        # The threshold is a cross-encoder score; RRF scores can't be compared to it.
        self.use(needs_rag=False, chunks=[self.relevant(5.0)])
        engine.answer("which model writes the answer?", options=engine.Options(rerank=False))
        self.assertEqual(self.retriever.queries, [])

    def test_sources_are_numbered_in_the_prompt_for_citation(self):
        self.use(needs_rag=True, chunks=[chunk(0), chunk(1)])
        engine.answer("what does the readme say")
        prompt = self.sent()["messages"][-1]["content"]
        self.assertIn("[1] doc0.md", prompt)
        self.assertIn("[2] doc1.md", prompt)

    def test_progress_is_reported_before_the_first_token(self):
        self.use(chain=("a", "b"), scripts={"a": ConnectionError("down"), "b": stream("ok")})
        seen = []
        engine.answer("hello", on_status=seen.append)
        self.assertEqual(seen[0], "Searching your documents…")
        self.assertTrue(seen[1].startswith("Loading a"))
        self.assertTrue(seen[2].startswith("Loading b"))  # the fallback is announced too

    def test_applied_action_has_the_full_result_shape(self):
        with mock.patch.object(engine.pc_agent, "apply", return_value={"answer": "done"}):
            result = engine.apply_pending({"op": "organize", "path": "x"})
        self.assertEqual(result["answer"], "done")
        self.assertEqual(set(result), set(engine._empty_result("x")))

    def test_system_agent_only_previews(self):
        self.use(task="system_agent")
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "empty").mkdir()
            result = engine.answer(f'remove empty folders in "{d}"')
            self.assertTrue(result["requires_confirmation"])
            self.assertTrue((Path(d) / "empty").is_dir())
        self.assertEqual(store.recent_turns(1)[0]["model"], "pc-toolkit")


if __name__ == "__main__":
    unittest.main()
