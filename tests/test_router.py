import unittest

from router import TaskRouter


class TaskRouterTests(unittest.TestCase):
    def setUp(self):
        self.router = TaskRouter()

    def test_coding_query_routes_to_coding(self):
        decision = self.router.classify("Fix this Python TypeError and explain the cause")
        self.assertEqual(decision["task"], "coding")

    def test_coding_prefers_local_coder_when_installed(self):
        decision = self.router.route(
            "Fix this Python TypeError and explain the cause",
            available_models=["llama3.1:8b", "qwen2.5-coder:7b"],
        )
        self.assertEqual(decision["model"], "qwen2.5-coder:7b")

    def test_planning_prefers_the_local_reasoning_model(self):
        # A planning prompt should reach the chain-of-thought model rather
        # than the general chat model, even though it is slower to answer.
        decision = self.router.route(
            "Make me a step by step plan and roadmap to finish this project",
            available_models=["llama3.1:8b", "deepseek-r1:7b"],
        )
        self.assertEqual(decision["task"], "planning")
        self.assertEqual(decision["model"], "deepseek-r1:7b")

    def test_ollama_down_leaves_nothing_to_run(self):
        # With no local backend there is no cloud to fall through to any more.
        decision = self.router.route(
            "Write a python function to reverse a linked list",
            available_models=[],
        )
        self.assertEqual(decision["chain"], [])
        self.assertIsNone(decision["model"])

    def test_greeting_stays_general(self):
        self.assertEqual(self.router.classify("hi")["task"], "general")

    def test_no_backend_yields_empty_chain(self):
        decision = self.router.route("hello there", available_models=[])
        self.assertEqual(decision["chain"], [])
        self.assertIsNone(decision["model"])

    def test_reasoning_query_routes_to_reasoning(self):
        decision = self.router.classify("Compare the trade-offs between a cache, queue, and database for this design")
        self.assertEqual(decision["task"], "reasoning")

    def test_general_query_routes_to_general(self):
        decision = self.router.classify("Can you summarize the NEXUS project in plain English?")
        self.assertEqual(decision["task"], "general")

    def test_rag_query_requires_local_docs(self):
        decision = self.router.route("What does the project say about the local AI model map?")
        self.assertTrue(decision["needs_rag"])

    def test_system_agent_routes_to_system_agent(self):
        decision = self.router.classify("Find duplicate files taking up space in my folder")
        self.assertEqual(decision["task"], "system_agent")


if __name__ == "__main__":
    unittest.main()
