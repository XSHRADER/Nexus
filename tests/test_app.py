import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_TMP = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
os.environ["NEXUS_DB"] = os.path.join(_TMP.name, "app-test.db")

from streamlit.testing.v1 import AppTest  # noqa: E402

from nexus import (
    engine,  # noqa: E402
    providers,  # noqa: E402
    retrieve,  # noqa: E402
    store,  # noqa: E402
)

APP = str(Path(__file__).resolve().parent.parent / "app.py")
CHAIN = [
    {"model": "a", "provider": "ollama", "score": 1.0, "reason": "a: test"},
    {"model": "b", "provider": "ollama", "score": 0.9, "reason": "b: test"},
]


class FakeRetriever:
    def stats(self):
        return {"chunks": 0, "files": {}}

    def query(self, *args, **kwargs):
        return []


def tearDownModule():
    store.close()
    _TMP.cleanup()


class AppFlowTests(unittest.TestCase):
    def setUp(self):
        conn = store.connect()
        with conn:
            for table in ("messages", "chats", "turns"):
                conn.execute(f"DELETE FROM {table}")
        conn.close()
        self.calls = []
        for patcher in (
            mock.patch.object(engine, "answer", self.fake_answer),
            mock.patch.object(providers, "availability",
                              lambda *a, **k: {"ollama": ["a", "b"], "ollama_up": True,
                                               "loaded": []}),
            mock.patch.object(providers, "capabilities",
                              lambda m: {"caps": set(), "context_length": None,
                                         "parameter_size": None}),
            mock.patch.object(retrieve, "get_retriever", lambda: FakeRetriever()),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def fake_answer(self, question, base_dir=None, options=None, on_token=None,
                    on_thinking=None, history=None, chat_id=None):
        self.calls.append({"question": question, "options": options,
                           "history": history, "chat_id": chat_id})
        model = options.force_model if options and options.force_model else "a"
        if on_token:
            on_token(f"answer from {model}")
        return {
            "answer": f"answer from {model}", "model": model, "task": "general",
            "needs_rag": False, "sources": [], "elapsed": 0.1, "chain": CHAIN,
            "complexity": 0.1, "attempts": [{"model": model, "provider": "ollama", "error": None}],
            "prompt_chars": len(question), "info": None, "auto_task": "general",
            "thinking": None, "truncated": False, "chunks_dropped": 0, "metrics": {},
            "requires_confirmation": False, "pending": None,
        }

    def start(self):
        at = AppTest.from_file(APP, default_timeout=90)
        at.run()
        self.assertFalse(at.exception, at.exception)
        return at

    def ask(self, at, text):
        at.chat_input[0].set_value(text).run()
        self.assertFalse(at.exception, at.exception)

    def test_answer_is_saved_and_reopens_in_a_new_session(self):
        at = self.start()
        self.ask(at, "what is hybrid retrieval?")
        [chat] = store.list_chats()
        self.assertEqual(chat["title"], "what is hybrid retrieval?")
        self.assertEqual([m["role"] for m in store.load_messages(chat["id"])],
                         ["user", "assistant"])

        fresh = self.start()
        self.assertEqual(len(fresh.chat_message), 0)
        fresh.button(key=f"open_{chat['id']}").click().run()
        self.assertEqual(len(fresh.chat_message), 2)

    def test_follow_up_sends_the_earlier_turns(self):
        at = self.start()
        self.ask(at, "first question")
        self.ask(at, "and the second one?")
        last = self.calls[-1]
        self.assertEqual(last["history"], [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "answer from a"},
        ])
        self.assertEqual(last["chat_id"], store.list_chats()[0]["id"])

    def test_regenerate_hides_the_answer_being_redone(self):
        at = self.start()
        self.ask(at, "only question")
        at.button(key="regen_b").click().run()
        last = self.calls[-1]
        self.assertEqual(last["options"].force_model, "b")
        self.assertEqual(last["history"], [])

    def test_deleting_a_chat(self):
        at = self.start()
        self.ask(at, "to be deleted")
        [chat] = store.list_chats()
        at.button(key=f"del_{chat['id']}").click().run()
        at.button(key="confirm_delete_chat").click().run()
        self.assertEqual(store.list_chats(), [])


if __name__ == "__main__":
    unittest.main()
