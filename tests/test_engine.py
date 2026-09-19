import os
import tempfile
import unittest
from unittest import mock

_TMP = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
os.environ["NEXUS_DB"] = os.path.join(_TMP.name, "engine-test.db")

import engine  # noqa: E402
from fakes import FakeOllama, stream  # noqa: E402

MESSAGES = [{"role": "user", "content": "hi"}]


class OllamaChatTests(unittest.TestCase):
    def use(self, scripts):
        self.ollama = FakeOllama(scripts)
        patcher = mock.patch.object(engine.requests, "post", self.ollama)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_streams_tokens_and_reports_metrics(self):
        self.use({"a": stream("Hel", "lo")})
        seen = []
        reply = engine._ollama_chat("a", MESSAGES, on_token=seen.append)
        self.assertEqual(reply.text, "Hello")
        self.assertEqual(seen, ["Hel", "lo"])
        self.assertIsNone(reply.thinking)
        self.assertEqual(reply.metrics["tokens_per_s"], 20.0)
        self.assertEqual(reply.metrics["load_ms"], 2000.0)
        self.assertEqual(reply.metrics["prompt_tokens"], 50)
        self.assertEqual(reply.metrics["eval_tokens"], 20)
        self.assertIsNotNone(reply.metrics["ttft_ms"])

    def test_payload_uses_chat_endpoint_window_and_timeouts(self):
        self.use({"a": stream("ok")})
        engine._ollama_chat("a", MESSAGES, temperature=0.2, num_ctx=4096)
        call = self.ollama.calls[0]
        self.assertTrue(call["url"].endswith("/api/chat"))
        self.assertTrue(call["stream"])
        self.assertEqual(call["timeout"], (5, 120))
        self.assertEqual(call["json"]["messages"], MESSAGES)
        self.assertEqual(call["json"]["options"], {"temperature": 0.2, "num_ctx": 4096})
        self.assertNotIn("think", call["json"])

    def test_think_flag_is_sent_only_when_asked(self):
        self.use({"a": stream("ok")})
        engine._ollama_chat("a", MESSAGES, think=True)
        self.assertIs(self.ollama.calls[0]["json"]["think"], True)

    def test_native_thinking_is_kept_separate(self):
        self.use({"a": stream("Answer", thinking=("step one ", "step two"))})
        thoughts = []
        reply = engine._ollama_chat("a", MESSAGES, on_thinking=thoughts.append)
        self.assertEqual(reply.text, "Answer")
        self.assertEqual(reply.thinking, "step one step two")
        self.assertEqual(thoughts, ["step one ", "step two"])

    def test_inline_think_block_is_moved_out_of_the_answer(self):
        self.use({"a": stream("<think>plan it</think>", "\n\nDone")})
        reply = engine._ollama_chat("a", MESSAGES)
        self.assertEqual(reply.text, "Done")
        self.assertEqual(reply.thinking, "plan it")

    def test_close_tag_only(self):
        self.use({"a": stream("plan it</think>Done")})
        reply = engine._ollama_chat("a", MESSAGES)
        self.assertEqual((reply.text, reply.thinking), ("Done", "plan it"))

    def test_error_chunk_raises(self):
        self.use({"a": [{"error": "model not found"}]})
        with self.assertRaisesRegex(RuntimeError, "model not found"):
            engine._ollama_chat("a", MESSAGES)

    def test_empty_reply_raises(self):
        self.use({"a": stream()})
        with self.assertRaisesRegex(RuntimeError, "empty"):
            engine._ollama_chat("a", MESSAGES)

    def test_guard_wraps_callback_errors(self):
        def boom(_):
            raise ValueError("ui gone")

        with self.assertRaises(engine._CallbackRaised) as caught:
            engine._guard(boom)("x")
        self.assertIsInstance(caught.exception.original, ValueError)
        self.assertIsNone(engine._guard(None))


if __name__ == "__main__":
    unittest.main()
