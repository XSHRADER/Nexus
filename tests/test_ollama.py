import os
import unittest
from unittest import mock

import isolate  # noqa: F401  (must precede any nexus import)
from fakes import FakeOllama, FakeResponse, stream

from nexus import config, ollama

MESSAGES = [{"role": "user", "content": "hi"}]


class OllamaChatTests(unittest.TestCase):
    def use(self, scripts):
        self.ollama = FakeOllama(scripts)
        patcher = mock.patch.object(ollama.requests, "post", self.ollama)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_streams_tokens_and_reports_metrics(self):
        self.use({"a": stream("Hel", "lo")})
        seen = []
        reply = ollama.chat("a", MESSAGES, on_token=seen.append)
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
        ollama.chat("a", MESSAGES, temperature=0.2, num_ctx=4096)
        call = self.ollama.calls[0]
        self.assertTrue(call["url"].endswith("/api/chat"))
        self.assertTrue(call["stream"])
        self.assertEqual(call["timeout"], (5, 120))
        self.assertEqual(call["json"]["messages"], MESSAGES)
        self.assertEqual(call["json"]["options"], {"temperature": 0.2, "num_ctx": 4096})
        self.assertNotIn("think", call["json"])

    def test_think_flag_is_sent_only_when_asked(self):
        self.use({"a": stream("ok")})
        ollama.chat("a", MESSAGES, think=True)
        self.assertIs(self.ollama.calls[0]["json"]["think"], True)

    def test_native_thinking_is_kept_separate(self):
        self.use({"a": stream("Answer", thinking=("step one ", "step two"))})
        thoughts = []
        reply = ollama.chat("a", MESSAGES, on_thinking=thoughts.append)
        self.assertEqual(reply.text, "Answer")
        self.assertEqual(reply.thinking, "step one step two")
        self.assertEqual(thoughts, ["step one ", "step two"])

    def test_inline_think_block_is_moved_out_of_the_answer(self):
        self.use({"a": stream("<think>plan it</think>", "\n\nDone")})
        reply = ollama.chat("a", MESSAGES)
        self.assertEqual(reply.text, "Done")
        self.assertEqual(reply.thinking, "plan it")

    def test_close_tag_only(self):
        self.use({"a": stream("plan it</think>Done")})
        reply = ollama.chat("a", MESSAGES)
        self.assertEqual((reply.text, reply.thinking), ("Done", "plan it"))

    def test_error_chunk_raises(self):
        self.use({"a": [{"error": "model not found"}]})
        with self.assertRaisesRegex(RuntimeError, "model not found"):
            ollama.chat("a", MESSAGES)

    def test_empty_reply_raises(self):
        self.use({"a": stream()})
        with self.assertRaisesRegex(RuntimeError, "empty"):
            ollama.chat("a", MESSAGES)

    def test_http_error_carries_ollamas_own_message(self):
        # "HTTP 404" alone doesn't say the model was never pulled.
        def post(url, json=None, **_):
            return FakeResponse([{"error": "model 'a' not found"}], status=404)

        with mock.patch.object(ollama.requests, "post", post),                 self.assertRaisesRegex(RuntimeError, "model 'a' not found"):
            ollama.chat("a", MESSAGES)


class ListingTests(unittest.TestCase):
    def setUp(self):
        ollama.clear_cache()
        self.addCleanup(ollama.clear_cache)

    def test_down_daemon_is_cached_as_unreachable(self):
        # A stopped Ollama used to cost a refused connection on every rerun.
        with mock.patch.object(
            ollama.requests, "get", side_effect=ollama.requests.ConnectionError("down")
        ) as get:
            self.assertEqual(ollama.installed_models(), [])
            self.assertFalse(ollama.reachable())
        self.assertEqual(get.call_count, 1)

    def test_lists_model_names(self):
        response = mock.Mock(status_code=200)
        response.json.return_value = {"models": [{"name": "llama3.1:8b"}, {"name": "qwen2.5:7b"}]}
        with mock.patch.object(ollama.requests, "get", return_value=response):
            self.assertEqual(ollama.installed_models(), ["llama3.1:8b", "qwen2.5:7b"])


class OllamaUrlTests(unittest.TestCase):
    def url(self, **env):
        keys = {"NEXUS_OLLAMA_URL": None, "OLLAMA_HOST": None, **env}
        with mock.patch.dict(os.environ, {k: v for k, v in keys.items() if v is not None}, clear=False):
            for k, v in keys.items():
                if v is None:
                    os.environ.pop(k, None)
            return config._ollama_url()

    def test_default_is_ipv4_loopback(self):
        # "localhost" can resolve to ::1 first on Windows while Ollama only
        # listens on 127.0.0.1, adding a connect timeout to every call.
        self.assertEqual(self.url(), "http://127.0.0.1:11434")

    def test_ollama_bind_address_is_turned_into_a_client_address(self):
        self.assertEqual(self.url(OLLAMA_HOST="0.0.0.0"), "http://127.0.0.1:11434")
        self.assertEqual(self.url(OLLAMA_HOST="0.0.0.0:9999"), "http://127.0.0.1:9999")

    def test_explicit_url_wins(self):
        self.assertEqual(
            self.url(NEXUS_OLLAMA_URL="http://gpu-box:11434/", OLLAMA_HOST="0.0.0.0"),
            "http://gpu-box:11434",
        )

    def test_host_without_scheme(self):
        self.assertEqual(self.url(OLLAMA_HOST="192.168.1.5"), "http://192.168.1.5:11434")


class IsolationTests(unittest.TestCase):
    def test_the_suite_never_touches_real_user_data(self):
        root = os.path.realpath(os.environ["NEXUS_TEST_ROOT"])
        for path in (config.DATA_DIR, config.DOCS_DIR, config.INDEX_DIR,
                     config.ROUTER_LOG, config.APP_LOG):
            self.assertTrue(os.path.realpath(path).startswith(root), path)


if __name__ == "__main__":
    unittest.main()
