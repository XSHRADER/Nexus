import unittest
from unittest import mock

from nexus import ollama, providers


class ComplexityTests(unittest.TestCase):
    def test_greeting_is_simple(self):
        self.assertLess(providers.estimate_complexity("hi"), 0.1)

    def test_planning_language_raises_complexity(self):
        simple = providers.estimate_complexity("what is a queue")
        planned = providers.estimate_complexity(
            "give me a step by step plan for migrating this, and outline the milestones"
        )
        self.assertGreater(planned, simple + 0.3)

    def test_empty_query(self):
        self.assertEqual(providers.estimate_complexity("   "), 0.0)


class ResolveInstalledTests(unittest.TestCase):
    def test_exact_tag(self):
        spec = providers.BY_NAME["llama3.1:8b"]
        self.assertEqual(providers.resolve_installed(spec, ["llama3.1:8b"]), "llama3.1:8b")

    def test_latest_tag_still_matches(self):
        spec = providers.BY_NAME["llama3.1:8b"]
        self.assertEqual(providers.resolve_installed(spec, ["llama3.1:latest"]), "llama3.1:latest")

    def test_quantised_tag_of_the_same_size_matches(self):
        spec = providers.BY_NAME["llama3.1:8b"]
        tag = "llama3.1:8b-instruct-q4_K_M"
        self.assertEqual(providers.resolve_installed(spec, [tag]), tag)

    def test_a_different_size_is_a_different_model(self):
        spec = providers.BY_NAME["llama3.1:8b"]
        self.assertIsNone(providers.resolve_installed(spec, ["llama3.1:70b"]))

    def test_missing_model(self):
        spec = providers.BY_NAME["llama3.1:8b"]
        self.assertIsNone(providers.resolve_installed(spec, ["mistral:7b"]))


class PlanTests(unittest.TestCase):
    AVAIL = {"ollama": ["llama3.1:8b", "qwen2.5-coder:7b", "deepseek-r1:7b", "llava:7b"]}

    def _top(self, task, query, avail=None):
        chain = providers.plan(task, query, avail=avail or self.AVAIL)
        return chain[0].model if chain else None

    def test_planning_goes_to_the_reasoning_specialist(self):
        # deepseek-r1 is the only installed model with real planning strength,
        # so it must outrank the general chat model despite being slower.
        self.assertEqual(
            self._top("planning", "draft a roadmap with milestones for the next month"),
            "deepseek-r1:7b",
        )

    def test_everyday_chat_stays_local(self):
        self.assertEqual(self._top("general", "what is a vector database"), "llama3.1:8b")

    def test_coding_prefers_specialist_local_model(self):
        self.assertEqual(self._top("coding", "write a function to sort a list"), "qwen2.5-coder:7b")

    def test_vision_models_are_never_offered_for_text(self):
        for task in providers.TEXT_TASKS:
            chain = providers.plan(task, "what is in this image", avail=self.AVAIL)
            self.assertNotIn("llava:7b", [c.model for c in chain], task)

    def test_each_installed_model_appears_once(self):
        # Two sizes of one family both resolve to a bare `qwen2.5` tag; the
        # engine must not try the same model twice.
        catalog = [
            providers.ModelSpec("qwen2.5:7b", "ollama", {"general": 0.8}),
            providers.ModelSpec("qwen2.5:14b", "ollama", {"general": 0.9}),
        ]
        with mock.patch.object(providers, "CATALOG", catalog):
            chain = providers.plan("general", "hello", avail={"ollama": ["qwen2.5"]})
        self.assertEqual([c.model for c in chain], ["qwen2.5"])
        self.assertEqual(chain[0].spec.name, "qwen2.5:14b")  # the better fit wins

    def test_chain_is_ordered_best_first(self):
        chain = providers.plan("reasoning", "compare these trade-offs", avail=self.AVAIL)
        scores = [c.score for c in chain]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_nothing_reachable(self):
        chain = providers.plan("general", "hello", avail={"ollama": []})
        self.assertEqual(chain, [])


class WarmModelTests(unittest.TestCase):
    INSTALLED = ["llama3.1:8b", "qwen2.5:7b", "mistral:7b", "qwen2.5-coder:7b"]

    def _avail(self, loaded):
        return providers.availability(self.INSTALLED, loaded)

    def test_resident_model_is_preferred_over_an_equal_peer(self):
        # mistral and qwen2.5 are close on general chat; the one already in
        # VRAM should win, since loading the other costs ~30s.
        cold = providers.plan("general", "tell me about caching",
                              avail=self._avail([]))
        warm = providers.plan("general", "tell me about caching",
                              avail=self._avail(["mistral:7b"]))
        cold_rank = [c.model for c in cold].index("mistral:7b")
        warm_rank = [c.model for c in warm].index("mistral:7b")
        self.assertLess(warm_rank, cold_rank)

    def test_warm_bonus_does_not_beat_a_specialist(self):
        # A general model sitting in VRAM must not steal a coding task from
        # the dedicated coder model.
        chain = providers.plan("coding", "fix this TypeError",
                               avail=self._avail(["mistral:7b"]))
        self.assertEqual(chain[0].model, "qwen2.5-coder:7b")


NO_CAPS = {"caps": set(), "context_length": None, "parameter_size": None}


class DiscoveryTests(unittest.TestCase):
    CAPS = {
        "qwen3:8b": {"completion", "thinking", "tools"},
        "bge-m3:latest": {"embedding"},
    }

    def setUp(self):
        patcher = mock.patch.object(
            providers, "capabilities",
            lambda m: {**NO_CAPS, "caps": self.CAPS.get(m, set())},
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _one(self, tag):
        specs = providers.discover([tag])
        self.assertEqual(len(specs), 1, specs)
        return specs[0]

    def test_unknown_general_model_gets_a_scaled_general_profile(self):
        spec = self._one("granite3.3:8b")
        self.assertTrue(spec.discovered)
        self.assertAlmostEqual(spec.strengths["general"], 0.75 * 0.9)
        self.assertEqual((spec.quality, spec.speed), (0.58, 0.80))

    def test_catalogued_models_are_not_rediscovered(self):
        self.assertEqual(providers.discover(["llama3.1:8b", "qwen2.5-coder:7b"]), [])

    def test_embedding_models_are_excluded(self):
        self.assertEqual(providers.discover(["nomic-embed-text:latest", "bge-m3:latest"]), [])

    def test_coder_family_and_size_from_the_tag(self):
        spec = self._one("qwen3-coder:30b")
        self.assertAlmostEqual(spec.strengths["coding"], 0.80 * 0.9)
        self.assertEqual((spec.quality, spec.speed), (0.75, 0.25))

    def test_reasoner_by_name(self):
        spec = self._one("qwq:32b")
        self.assertEqual(max(spec.strengths, key=spec.strengths.get), "reasoning")

    def test_thinking_capability_extends_a_general_model(self):
        spec = self._one("qwen3:8b")
        self.assertAlmostEqual(spec.strengths["general"], 0.75 * 0.9)
        self.assertAlmostEqual(spec.strengths["reasoning"], 0.78 * 0.9)

    def test_vision_by_name_is_vision_only(self):
        spec = self._one("llama3.2-vision:11b")
        self.assertEqual(set(spec.strengths), {"vision"})

    def test_catalogue_entry_beats_a_discovered_peer(self):
        chain = providers.plan(
            "general", "what is a vector database",
            avail={"ollama": ["llama3.1:8b", "granite3.3:8b"]},
        )
        self.assertEqual(chain[0].model, "llama3.1:8b")
        discovered = next(c for c in chain if c.model == "granite3.3:8b")
        self.assertIn("inferred", discovered.reason)

    def test_discovered_model_answers_when_nothing_catalogued_is_installed(self):
        chain = providers.plan("general", "hello there", avail={"ollama": ["granite3.3:8b"]})
        self.assertEqual(chain[0].model, "granite3.3:8b")


class CapabilitiesTests(unittest.TestCase):
    def setUp(self):
        providers._CAPS_CACHE.clear()
        self.addCleanup(providers._CAPS_CACHE.clear)

    def test_parses_show_and_caches(self):
        response = mock.Mock(status_code=200)
        response.json.return_value = {
            "capabilities": ["completion", "Tools"],
            "details": {"parameter_size": "8.0B"},
            "model_info": {"llama.context_length": 131072},
        }
        with mock.patch.object(ollama.requests, "post", return_value=response) as post:
            info = providers.capabilities("llama3.1:8b")
            providers.capabilities("llama3.1:8b")
        self.assertEqual(info["caps"], {"completion", "tools"})
        self.assertEqual(info["context_length"], 131072)
        self.assertEqual(info["parameter_size"], 8.0)
        self.assertEqual(post.call_count, 1)

    def test_degrades_and_retries_when_ollama_is_down(self):
        with mock.patch.object(
            ollama.requests, "post",
            side_effect=ollama.requests.ConnectionError("down"),
        ) as post:
            info = providers.capabilities("llama3.1:8b")
            providers.capabilities("llama3.1:8b")
        self.assertEqual(info, NO_CAPS)
        self.assertEqual(post.call_count, 2)


if __name__ == "__main__":
    unittest.main()
