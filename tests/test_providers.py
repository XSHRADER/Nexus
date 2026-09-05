import unittest

import providers


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

    def test_missing_model(self):
        spec = providers.BY_NAME["llama3.1:8b"]
        self.assertIsNone(providers.resolve_installed(spec, ["mistral:7b"]))


class PlanTests(unittest.TestCase):
    BOTH = {"ollama": ["llama3.1:8b", "qwen2.5-coder:7b", "deepseek-r1:7b", "llava:7b"],
            "gemini": True}

    def _top(self, task, query, needs_rag=False, avail=None):
        chain = providers.plan(task, query, needs_rag, avail=avail or self.BOTH)
        return chain[0].model if chain else None

    def test_planning_goes_to_gemini_pro(self):
        self.assertEqual(
            self._top("planning", "draft a roadmap with milestones for the next month"),
            "gemini-2.5-pro",
        )

    def test_everyday_chat_stays_local(self):
        self.assertEqual(self._top("general", "what is a vector database"), "llama3.1:8b")

    def test_coding_prefers_specialist_local_model(self):
        self.assertEqual(self._top("coding", "write a function to sort a list"), "qwen2.5-coder:7b")

    def test_rag_keeps_documents_local(self):
        chain = providers.plan("general", "summarize my notes", needs_rag=True, avail=self.BOTH)
        self.assertTrue(chain[0].spec.is_local)

    def test_only_capable_models_are_offered(self):
        chain = providers.plan("vision", "what is in this image", avail=self.BOTH)
        for candidate in chain:
            self.assertIn("vision", candidate.spec.strengths)

    def test_chain_is_ordered_best_first(self):
        chain = providers.plan("reasoning", "compare these trade-offs", avail=self.BOTH)
        scores = [c.score for c in chain]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_nothing_reachable(self):
        chain = providers.plan("general", "hello", avail={"ollama": [], "gemini": False})
        self.assertEqual(chain, [])


class WarmModelTests(unittest.TestCase):
    INSTALLED = ["llama3.1:8b", "qwen2.5:7b", "mistral:7b", "qwen2.5-coder:7b"]

    def _avail(self, loaded):
        return providers.availability(self.INSTALLED, False, loaded)

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

    def test_cloud_models_are_never_warm(self):
        avail = providers.availability(self.INSTALLED, True, ["mistral:7b"])
        chain = providers.plan("planning", "draft a roadmap", avail=avail)
        self.assertEqual(chain[0].model, "gemini-2.5-pro")


if __name__ == "__main__":
    unittest.main()
