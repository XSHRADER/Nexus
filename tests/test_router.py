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


class IntentPrecisionTests(unittest.TestCase):
    """Keyword rules must need an action, not just a technology noun.

    Bare nouns (`python`, `api`, `class`) used to score full coding intent, so
    "what is an API?" was answered by the coder model; bare `folder` did the
    same for file operations, turning questions *about* the project into
    requests to act on disk.
    """

    @classmethod
    def setUpClass(cls):
        cls.router = TaskRouter()

    def assert_task(self, query, expected):
        got = self.router.classify(query)["task"]
        self.assertEqual(got, expected, f"{query!r} routed to {got}, expected {expected}")

    # -- nouns alone must not mean "coding" --------------------------------
    def test_version_question_is_general(self):
        self.assert_task("what Python version is used?", "general")

    def test_definition_question_is_general(self):
        self.assert_task("what is an API?", "general")

    def test_conceptual_question_is_general(self):
        self.assert_task("what does the class hierarchy look like conceptually?", "general")

    # -- but real coding requests still route to coding --------------------
    def test_fix_request_is_coding(self):
        self.assert_task("fix this TypeError in my function", "coding")

    def test_write_function_is_coding(self):
        self.assert_task("write a python function to reverse a list", "coding")

    def test_refactor_is_coding(self):
        self.assert_task("refactor this class to use dependency injection", "coding")

    def test_debug_is_coding(self):
        self.assert_task("debug why my script crashes on startup", "coding")

    def test_failing_test_is_coding(self):
        self.assert_task("my unit test is failing with an exception", "coding")

    def test_write_query_is_coding(self):
        self.assert_task("write a SQL query to join two tables", "coding")

    # -- file operations need a verb ---------------------------------------
    def test_folder_question_is_not_a_file_operation(self):
        self.assert_task("explain the folder structure of this project", "general")

    def test_sort_request_is_system_agent(self):
        self.assert_task("sort my Downloads folder", "system_agent")

    def test_cleanup_request_is_system_agent(self):
        self.assert_task("clean up empty folders", "system_agent")

    def test_undo_request_is_system_agent(self):
        self.assert_task("undo the last folder organization", "system_agent")

    # -- plural spelling used to fall through ------------------------------
    def test_plural_tradeoffs_is_reasoning(self):
        # `tradeoff\b` never matched "tradeoffs", which is how most people
        # actually write it, so these fell back to general chat.
        self.assert_task("what are the tradeoffs of local vs cloud inference?", "reasoning")

    def test_singular_tradeoff_still_reasoning(self):
        self.assert_task("compare the trade-off between cache and queue", "reasoning")


if __name__ == "__main__":
    unittest.main()
