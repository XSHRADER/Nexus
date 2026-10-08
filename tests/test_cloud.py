"""Cloud providers (Phase 1).

End to end against two in-process fakes -- demos/mock_ollama.py for local
models and demos/mock_cloud.py for every cloud provider -- so routing,
privacy rules, fallbacks and the HTTP clients are all exercised for real,
with no network and no API keys.
"""

import json
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demos"))

import cloud  # noqa: E402
import cloud_client  # noqa: E402
import config  # noqa: E402
import engine  # noqa: E402
import providers  # noqa: E402
import speech  # noqa: E402
import store  # noqa: E402
import mock_cloud  # noqa: E402
import mock_ollama  # noqa: E402

KEYS = {
    "GEMINI_API_KEY": "k-gemini",
    "GROQ_API_KEY": "k-groq",
    "OPENROUTER_API_KEY": "k-openrouter",
    "DEEPSEEK_API_KEY": "k-deepseek",
    "MISTRAL_API_KEY": "k-mistral",
}
NO_KEYS = {k: "" for k in KEYS}

EASY = "what is a queue"
HARD = ("Make me a step by step plan and roadmap to migrate our monolith, compare the "
        "trade-offs of each approach, outline the milestones and justify the order.")


class FakeRouter:
    """Classifies by the test's say-so, but plans with the real providers.plan()."""

    def __init__(self, task="general", complexity=None, needs_rag=False, installed=None):
        self.task, self.complexity, self.needs_rag = task, complexity, needs_rag
        self.installed = installed if installed is not None else ["llama3.1:8b",
                                                                  "deepseek-r1:7b"]

    def route(self, question, force_task=None, policy=None, needs_image=False,
              rag_override=None):
        task = force_task or self.task
        needs_rag = self.needs_rag if rag_override is None else rag_override
        complexity = (providers.estimate_complexity(question)
                      if self.complexity is None else self.complexity)
        avail = providers.availability(self.installed, [])
        chain = [
            {"model": c.model, "provider": c.spec.provider, "local": c.spec.is_local,
             "score": c.score, "reason": c.reason}
            for c in providers.plan(task, question, needs_rag, complexity, avail,
                                    policy=policy, needs_image=needs_image)
        ]
        return {"task": task, "model": chain[0]["model"] if chain else None,
                "provider": chain[0]["provider"] if chain else None, "chain": chain,
                "complexity": complexity, "available": avail, "needs_rag": needs_rag,
                "confidence": 1.0, "scores": {task: 1.0}, "reason": "test"}


class CloudTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ollama_server, cls.ollama_url = mock_ollama.start_in_thread()
        cls.cloud_server, cls.cloud_url = mock_cloud.start_in_thread()

    @classmethod
    def tearDownClass(cls):
        cls.ollama_server.shutdown()
        cls.cloud_server.shutdown()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = store.ChatStore(Path(self.tmp.name) / "t.db")
        self._saved_store = store._default
        store._default = self.store
        mock_cloud.reset()
        cloud.reset_state()
        self.env = mock.patch.dict(os.environ, KEYS)
        self.env.start()
        self.ollama_patch = mock.patch.object(engine, "OLLAMA_CHAT_URL",
                                              f"{self.ollama_url}/api/chat")
        self.ollama_patch.start()

    def tearDown(self):
        self.ollama_patch.stop()
        self.env.stop()
        store._default = self._saved_store
        cloud.reset_state()
        cloud.cloud_specs.cache_clear()
        self.tmp.cleanup()

    @contextmanager
    def settings(self, **overrides):
        """Settings pointing every provider at the mock cloud."""
        base = config.Settings(
            daily_limits=dict(config.DEFAULT_DAILY_LIMITS),
            provider_overrides={
                p: {"base_url": f"{self.cloud_url}/{p}/v1"}
                for p in cloud.PROVIDERS
            },
        )
        s = replace(base, **overrides)
        patches = [mock.patch.object(m, "get_settings", return_value=s)
                   for m in (cloud, engine, speech)]
        for p in patches:
            p.start()
        cloud.cloud_specs.cache_clear()
        try:
            yield s
        finally:
            for p in patches:
                p.stop()
            cloud.cloud_specs.cache_clear()

    def ask(self, question, router=None, options=None, **kwargs):
        with mock.patch.object(engine, "get_router", return_value=router or FakeRouter()):
            return engine.answer(question, options=options or engine.Options(), **kwargs)

    def chain_models(self, router, question, options):
        policy = cloud.CloudPolicy.from_settings(options.cloud_mode, options.allow_docs,
                                                 options.allow_paid)
        return [c["model"] for c in router.route(question, policy=policy)["chain"]]


class PolicyTests(CloudTestCase):
    def test_no_key_means_no_cloud_model(self):
        with self.settings(), mock.patch.dict(os.environ, NO_KEYS):
            chain = self.chain_models(FakeRouter(complexity=0.9), HARD,
                                      engine.Options(cloud_mode="allowed", allow_paid=True))
        self.assertEqual(set(chain), {"llama3.1:8b", "deepseek-r1:7b"})

    def test_cloud_off_matches_local_only_routing(self):
        with self.settings():
            off = self.chain_models(FakeRouter(complexity=0.9), HARD,
                                    engine.Options(cloud_mode="off"))
            local = [c.model for c in providers.plan(
                "general", HARD, False, 0.9,
                {"ollama": ["llama3.1:8b", "deepseek-r1:7b"], "loaded": []},
                policy=cloud.CloudPolicy(mode="off"))]
        self.assertEqual(off, local)
        self.assertTrue(all(":" in m for m in off))  # Ollama tags only

    def test_default_settings_keep_cloud_off(self):
        with self.settings():
            chain = self.chain_models(FakeRouter(complexity=0.9), HARD, engine.Options())
        self.assertEqual(set(chain), {"llama3.1:8b", "deepseek-r1:7b"})

    def test_hard_mode_cloud_only_for_hard_prompts(self):
        with self.settings():
            easy = self.chain_models(FakeRouter(complexity=0.1), EASY,
                                     engine.Options(cloud_mode="hard"))
            hard = self.chain_models(FakeRouter(task="planning", complexity=0.9), HARD,
                                     engine.Options(cloud_mode="hard"))
        self.assertEqual(set(easy), {"llama3.1:8b", "deepseek-r1:7b"})
        self.assertFalse(providers.spec_by_name(hard[0]).is_local, hard)

    def test_allowed_mode_keeps_easy_prompts_local_first(self):
        with self.settings():
            chain = self.chain_models(FakeRouter(complexity=0.1), EASY,
                                      engine.Options(cloud_mode="allowed"))
        self.assertTrue(providers.spec_by_name(chain[0]).is_local)
        self.assertTrue(any(not providers.spec_by_name(m).is_local for m in chain))

    def test_paid_models_need_allow_paid(self):
        with self.settings():
            without = self.chain_models(FakeRouter(task="coding", complexity=0.9), HARD,
                                        engine.Options(cloud_mode="allowed"))
            with_paid = self.chain_models(FakeRouter(task="coding", complexity=0.9), HARD,
                                          engine.Options(cloud_mode="allowed", allow_paid=True))
        self.assertNotIn("openrouter/auto", without)
        self.assertIn("openrouter/auto", with_paid)

    def test_document_questions_stay_local_unless_allowed(self):
        router = FakeRouter(complexity=0.9, needs_rag=True)
        with self.settings():
            blocked = self.chain_models(router, HARD, engine.Options(cloud_mode="allowed"))
            allowed = self.chain_models(router, HARD,
                                        engine.Options(cloud_mode="allowed", allow_docs=True))
        self.assertTrue(all(providers.spec_by_name(m).is_local for m in blocked))
        self.assertTrue(any(not providers.spec_by_name(m).is_local for m in allowed))

    def test_pc_actions_never_go_to_cloud(self):
        spec = providers.spec_by_name("gemini-3.7-flash")
        policy = cloud.CloudPolicy(mode="allowed", allow_docs=True, allow_paid=True)
        self.assertIsNotNone(cloud.privacy_block(spec, "system_agent", False, policy))

    def test_daily_limit_skips_provider(self):
        with self.settings(daily_limits={"groq": 2}):
            for _ in range(2):
                self.store.record_usage("groq")
            self.assertEqual(cloud.provider_status("groq")["status"], "limit_reached")
            chain = self.chain_models(FakeRouter(complexity=0.9), HARD,
                                      engine.Options(cloud_mode="allowed"))
        self.assertFalse(any(m.startswith("openai/gpt-oss") for m in chain))

    def test_image_needs_image_capable_models(self):
        with self.settings():
            policy = cloud.CloudPolicy(mode="hard")
            chain = providers.plan("vision", "describe this", False, 0.2,
                                   providers.availability(["llama3.1:8b"], []),
                                   policy=policy, needs_image=True)
        self.assertTrue(chain)
        self.assertTrue(all("image" in c.spec.modalities for c in chain))
        # "hard" mode still uses cloud for an easy prompt when nothing local can see.
        self.assertEqual(chain[0].model, "gemini-3.7-flash")


class EngineCloudTests(CloudTestCase):
    def test_hard_prompt_answered_in_cloud_and_counted(self):
        with self.settings():
            result = self.ask(HARD, FakeRouter(task="planning", complexity=0.9),
                              engine.Options(cloud_mode="hard"))
        self.assertFalse(result["local"])
        self.assertIn("mock", result["answer"])
        self.assertIn("Here is a plan", result["answer"])
        self.assertEqual(self.store.usage_today(result["provider"]), 1)

    def test_rate_limit_falls_back_and_cools_down(self):
        with self.settings():
            first = self.chain_models(FakeRouter(task="planning", complexity=0.9), HARD,
                                      engine.Options(cloud_mode="hard"))[0]
            provider = providers.spec_by_name(first).provider
            mock_cloud.STATE["fail"] = {provider: 429}
            result = self.ask(HARD, FakeRouter(task="planning", complexity=0.9),
                              engine.Options(cloud_mode="hard"))
            self.assertNotEqual(result["model"], first)
            self.assertIn("rate-limited", result["info"])
            self.assertEqual(cloud.provider_status(provider)["status"], "cooling_down")
            # Next request doesn't even try the cooling provider.
            chain = self.chain_models(FakeRouter(task="planning", complexity=0.9), HARD,
                                      engine.Options(cloud_mode="hard"))
        self.assertTrue(all(providers.spec_by_name(m).provider != provider for m in chain))

    def test_everything_cloud_down_local_answers(self):
        with self.settings():
            mock_cloud.STATE["fail"] = {p: 500 for p in cloud.PROVIDERS}
            result = self.ask(HARD, FakeRouter(task="planning", complexity=0.9),
                              engine.Options(cloud_mode="hard", allow_paid=True))
        self.assertTrue(result["local"])
        self.assertIn("unreachable", result["info"])

    def test_offline_network_error_falls_back(self):
        with self.settings(provider_overrides={
                p: {"base_url": "http://127.0.0.1:9/v1"} for p in cloud.PROVIDERS}):
            result = self.ask(HARD, FakeRouter(task="planning", complexity=0.9),
                              engine.Options(cloud_mode="hard"))
        self.assertTrue(result["local"])

    def test_bad_key_marks_provider_invalid(self):
        with self.settings(), mock.patch.dict(os.environ, {**NO_KEYS, "GEMINI_API_KEY": "bad-key"}):
            result = self.ask(HARD, FakeRouter(task="planning", complexity=0.9),
                              engine.Options(cloud_mode="hard"))
            self.assertTrue(result["local"])
            self.assertEqual(cloud.provider_status("gemini")["status"], "invalid_key")
            with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "new-key"}):
                self.assertEqual(cloud.provider_status("gemini")["status"], "ready")

    def test_unknown_model_is_skipped_next_time(self):
        with self.settings(extra_models=[{
                "name": "gemini-retired", "provider": "gemini", "cost": "free",
                "quality": 0.99, "speed": 0.99, "strengths": {"planning": 0.99}}]):
            result = self.ask(HARD, FakeRouter(task="planning", complexity=0.9),
                              engine.Options(cloud_mode="hard"))
            self.assertNotEqual(result["model"], "gemini-retired")
            self.assertTrue(cloud.is_missing("gemini", "gemini-retired"))

    def test_pinned_cloud_model_still_obeys_privacy(self):
        chunks = [{"text": "secret plan", "meta": {"source": "s.md"}, "score": 1.0}]
        with self.settings(), mock.patch.object(engine, "retrieve_chunks", return_value=chunks):
            result = self.ask("what is in my notes?", FakeRouter(needs_rag=True),
                              engine.Options(cloud_mode="allowed",
                                             force_model="gemini-3.7-flash"))
        self.assertTrue(result["local"])
        self.assertIn("stay on this PC", result["info"])
        sent = [r for r in mock_cloud.STATE["requests"] if r["provider"] == "gemini"]
        self.assertEqual(sent, [])

    def test_pinned_cloud_model_blocked_when_cloud_off(self):
        with self.settings():
            result = self.ask(HARD, FakeRouter(task="planning", complexity=0.9),
                              engine.Options(cloud_mode="off", force_model="gemini-3.7-flash"))
        self.assertTrue(result["local"])
        self.assertIn("cloud is switched off", result["info"])
        self.assertEqual(mock_cloud.STATE["requests"], [])

    def test_cloud_never_sees_document_turns_in_history(self):
        history = [
            {"role": "user", "content": "what does my secret file say?"},
            {"role": "assistant", "content": "It says the launch code is 1234.",
             "meta": {"needs_rag": True}},
            {"role": "user", "content": "thanks"},
            {"role": "assistant", "content": "You're welcome."},
        ]
        with self.settings():
            result = self.ask(HARD, FakeRouter(task="planning", complexity=0.9),
                              engine.Options(cloud_mode="hard"), history=history)
        self.assertFalse(result["local"])
        body = json.dumps(mock_cloud.STATE["requests"][-1]["body"])
        self.assertNotIn("1234", body)
        self.assertIn("thanks", body)

    def test_image_reaches_cloud_vision_model(self):
        img = {"data": "aGVsbG8=", "mime": "image/png"}
        with self.settings():
            result = self.ask("what is this?", FakeRouter(installed=["llama3.1:8b"]),
                              engine.Options(cloud_mode="hard"), images=[img])
        self.assertEqual(result["model"], "gemini-3.7-flash")
        self.assertIn("1 image", result["answer"])
        parts = mock_cloud.STATE["requests"][-1]["body"]["messages"][-1]["content"]
        self.assertEqual(parts[1]["image_url"]["url"], "data:image/png;base64,aGVsbG8=")

    def test_image_with_cloud_off_and_no_local_vision_explains(self):
        with self.settings():
            result = self.ask("what is this?", FakeRouter(installed=["llama3.1:8b"]),
                              engine.Options(cloud_mode="off"),
                              images=[{"data": "aGVsbG8=", "mime": "image/png"}])
        self.assertIn("vision model", result["answer"])

    def test_local_vision_gets_ollama_images_field(self):
        msgs = engine._with_ollama_images([{"role": "user", "content": "hi"}],
                                          [{"data": "QUJD", "mime": "image/png"}])
        self.assertEqual(msgs[-1]["images"], ["QUJD"])


class ClientTests(CloudTestCase):
    def test_sse_parser_handles_noise_and_done(self):
        lines = [b"", b": keep-alive", b"data: not json",
                 b'data: {"choices":[{"delta":{"content":"Hel"}}]}',
                 'data: {"choices":[{"delta":{}}]}',
                 b'data: {"choices":[{"delta":{"content":"lo"}}]}',
                 b"data: [DONE]",
                 b'data: {"choices":[{"delta":{"content":"ignored"}}]}']
        self.assertEqual("".join(cloud_client.iter_sse_text(lines)), "Hello")

    def test_streaming_and_plain_chat(self):
        with self.settings():
            tokens = []
            streamed = cloud_client.chat("groq", "openai/gpt-oss-20b",
                                         [{"role": "user", "content": "hi"}],
                                         on_token=tokens.append)
            plain = cloud_client.chat("groq", "openai/gpt-oss-20b",
                                      [{"role": "user", "content": "hi"}])
        self.assertEqual(streamed, "".join(tokens).strip())
        self.assertEqual(streamed, plain)

    def test_error_mapping(self):
        with self.settings():
            for status, kind in ((429, cloud.RateLimited), (500, cloud.Offline)):
                mock_cloud.STATE["fail"] = {"gemini": status}
                with self.assertRaises(kind):
                    cloud_client.chat("gemini", "gemini-3.7-flash",
                                      [{"role": "user", "content": "x"}])
            mock_cloud.STATE["fail"] = {}
            with self.assertRaises(cloud.ModelNotFound):
                cloud_client.chat("gemini", "nope", [{"role": "user", "content": "x"}])

    def test_list_models_and_verify(self):
        with self.settings():
            self.assertIn("gemini-3.7-flash", cloud_client.list_models("gemini"))
            report = cloud.verify_models()
        self.assertEqual(report["gemini"]["missing"], [])


class SpeechTests(CloudTestCase):
    def test_cloud_transcription(self):
        with self.settings():
            out = engine.transcribe(b"RIFF....", "q.wav", "audio/wav",
                                    engine.Options(cloud_mode="hard"))
        self.assertEqual(out["text"], mock_cloud.TRANSCRIPT)
        self.assertFalse(out["local"])
        self.assertEqual(self.store.usage_today("groq"), 1)

    def test_cloud_off_without_local_whisper_explains(self):
        with self.settings(), mock.patch.object(speech, "local_available", return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                engine.transcribe(b"RIFF....", options=engine.Options(cloud_mode="off"))
        self.assertIn("faster-whisper", str(ctx.exception))


class CatalogTests(unittest.TestCase):
    def test_catalog_file_is_valid(self):
        entries = cloud._load_catalog_entries()
        self.assertTrue(entries)
        for e in entries:
            self.assertIn(e["provider"], cloud.PROVIDERS, e)
            self.assertIn(e.get("cost"), ("free", "cheap", "paid"), e)
            self.assertTrue(set(e.get("strengths", {})) <= set(
                ["general", "coding", "reasoning", "planning", "vision", "speech"]), e)

    def test_toml_cloud_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "n.toml"
            path.write_text(
                '[cloud]\nmode = "hard"\nallow_paid = true\n'
                '[cloud.daily_limits]\ngroq = 7\n'
                '[cloud.providers.gemini]\nbase_url = "http://x/v1"\n'
                '[[cloud.models]]\nname = "m"\nprovider = "groq"\n'
                '[speech]\nlocal_model = "base"\n')
            s = config.load_settings(path, Path(tmp) / "none.env")
        self.assertEqual(s.cloud_mode, "hard")
        self.assertTrue(s.allow_paid)
        self.assertFalse(s.allow_docs_to_cloud)
        self.assertEqual(s.daily_limits["groq"], 7)
        self.assertEqual(s.daily_limits["gemini"], 200)
        self.assertEqual(s.provider_overrides["gemini"]["base_url"], "http://x/v1")
        self.assertEqual(s.extra_models[0]["name"], "m")
        self.assertEqual(s.local_whisper_model, "base")

    def test_bad_mode_falls_back_to_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "n.toml"
            path.write_text('[cloud]\nmode = "always"\n')
            self.assertEqual(config.load_settings(path, Path(tmp) / "e").cloud_mode, "off")


if __name__ == "__main__":
    unittest.main()
