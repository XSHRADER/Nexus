import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import isolate  # noqa: F401  (must precede any nexus import)

_TMP = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
os.environ["NEXUS_DB"] = os.path.join(_TMP.name, "app-test.db")

from streamlit.testing.v1 import AppTest  # noqa: E402

from nexus import (
    engine,  # noqa: E402
    feedback,  # noqa: E402
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
            for table in ("messages", "chats", "turns", "feedback", "battles", "signals"):
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
                    on_thinking=None, history=None, chat_id=None, on_status=None, **extra):
        self.calls.append({"question": question, "options": options,
                           "history": history, "chat_id": chat_id, **extra})
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

    def test_every_page_renders(self):
        at = self.start()
        for page in ("documents.py", "lab.py", "diagnostics.py", "inbox.py",
                     "leaderboard.py", "chat.py"):
            at.switch_page(f"app_pages/{page}").run()
            self.assertFalse(at.exception, f"{page}: {at.exception}")

    def test_answer_settings_survive_a_page_switch(self):
        # Widget state is dropped when a widget isn't rendered, so settings
        # used to reset whenever the user looked at another page.
        at = self.start()
        at.selectbox(key="opt_model").set_value("b").run()
        at.switch_page("app_pages/diagnostics.py").run()
        at.switch_page("app_pages/chat.py").run()
        self.ask(at, "hello")
        self.assertEqual(self.calls[-1]["options"].force_model, "b")

    def test_suggestion_asks_the_question(self):
        at = self.start()
        at.pills(key="suggestion").set_value(at.pills(key="suggestion").options[0]).run()
        self.assertFalse(at.exception, at.exception)
        self.assertEqual(self.calls[-1]["question"], "What does this project use to store embeddings?")

    def test_chat_search_filters_the_sidebar(self):
        at = self.start()
        self.ask(at, "alpha question")
        at.button(key="new_chat").click().run()
        self.ask(at, "beta question")
        at.text_input(key="chat_search").input("alpha").run()
        titles = [b.label for b in at.sidebar.button if b.key and b.key.startswith("open_")]
        self.assertEqual(titles, ["alpha question"])

    def test_file_action_waits_for_confirmation(self):
        pending = {"op": "organize", "path": "C:/tmp/demo"}
        real = self.fake_answer

        def previewing(question, **kwargs):
            result = real(question, **kwargs)
            result.update(task="system_agent", model="pc-toolkit", chain=[],
                          answer="Plan: 2 files", requires_confirmation=True, pending=pending)
            return result

        applied = []
        with mock.patch.object(engine, "answer", previewing), \
                mock.patch.object(engine, "apply_pending",
                                  lambda p: applied.append(p) or {"answer": "Moved 2 files."}):
            at = self.start()
            self.ask(at, "sort my demo folder")
            self.assertEqual(applied, [])  # nothing happens without the click
            at.button(key="apply_pending").click().run()
        self.assertEqual(applied, [pending])
        chat = store.list_chats()[0]
        self.assertEqual(store.load_messages(chat["id"])[-1]["content"], "Moved 2 files.")

    # -- screens that came with ratings, Arena, the council and cloud ----------

    def set_mode(self, at, mode):
        next(g for g in at.button_group if g.key == "opt_mode").set_value(mode).run()
        self.assertFalse(at.exception, at.exception)

    def test_thumbs_down_is_saved_against_the_answer(self):
        at = self.start()
        self.ask(at, "rate me")
        answer_id = store.load_messages(store.list_chats()[0]["id"])[-1]["id"]
        at.button(key=f"down_{answer_id}").click().run()
        self.assertFalse(at.exception, at.exception)
        [rating] = feedback.ratings()
        self.assertEqual((rating["message_id"], rating["rating"], rating["question"]),
                         (answer_id, -1, "rate me"))
        self.assertEqual(store.get_message(answer_id)["meta"]["rating"], -1)

    def test_wrong_task_correction_is_saved_for_training(self):
        at = self.start()
        self.ask(at, "why is my loop slow?")
        answer_id = store.load_messages(store.list_chats()[0]["id"])[-1]["id"]
        at.pills(key=f"correct_{answer_id}").set_value("coding").run()
        self.assertFalse(at.exception, at.exception)
        [signal] = feedback.signals("task_override")
        self.assertEqual((signal["question"], signal["task"], signal["value"]),
                         ("why is my loop slow?", "general", "coding"))
        self.assertEqual(store.get_message(answer_id)["meta"]["task_corrected"], "coding")
        # Asked once: the control is replaced by a note, not offered again.
        self.assertFalse([p for p in at.pills if p.key == f"correct_{answer_id}"])

    def test_arena_hides_the_models_until_the_vote(self):
        at = self.start()
        self.set_mode(at, "Arena")
        self.ask(at, "explain a hash table")
        asked = {c["options"].force_model for c in self.calls}
        self.assertEqual(asked, {"a", "b"})          # two different models answered
        shown = " ".join(m.value for m in at.markdown)
        self.assertIn("Answer A", shown)
        # Nothing but your question is in the chat yet: no model has been named.
        chat = store.list_chats()[0]
        self.assertEqual([m["role"] for m in store.load_messages(chat["id"])], ["user"])

        at.button(key="vote_a").click().run()
        self.assertFalse(at.exception, at.exception)
        [battle] = feedback.battles()
        self.assertEqual(battle["winner"], "a")
        saved = store.load_messages(chat["id"])[-1]
        self.assertEqual(saved["meta"]["model"], battle["model_a"])
        self.assertIn(battle["model_b"], saved["meta"]["info"])   # both names revealed now

    def test_council_saves_one_merged_answer_with_its_members(self):
        at = self.start()
        self.set_mode(at, "Council")
        with mock.patch.object(engine, "_generate", return_value="not json"):  # no usable judge
            self.ask(at, "explain a hash table")
        chat = store.list_chats()[0]
        saved = store.load_messages(chat["id"])[-1]
        self.assertEqual(saved["meta"]["model"], "council")
        self.assertEqual(sorted(m["model"] for m in saved["meta"]["council"]["members"]), ["a", "b"])
        self.assertTrue(saved["content"].startswith("answer from"))

    def test_cloud_stays_off_unless_switched_on(self):
        at = self.start()
        self.ask(at, "hello")
        self.assertEqual(self.calls[-1]["options"].cloud_mode, "off")
        next(g for g in at.button_group if g.key == "opt_cloud").set_value("Hard questions").run()
        self.ask(at, "hello again")
        options = self.calls[-1]["options"]
        self.assertEqual(options.cloud_mode, "hard")
        self.assertFalse(options.allow_docs)   # documents still stay on this PC
        self.assertFalse(options.allow_paid)

    def test_paid_cloud_models_are_not_offered_until_allowed(self):
        ready = {"status": "ready", "detail": "0/50 requests today", "used": 0, "limit": 50}
        avail = {"ollama": ["a", "b"], "ollama_up": True, "loaded": [], "cloud": {"openrouter": ready}}
        from nexus import ui

        with mock.patch.object(providers, "availability", lambda *a, **k: avail):
            ui.availability.clear()   # cached for a few seconds, across sessions
            self.addCleanup(ui.availability.clear)
            at = self.start()
            self.assertEqual(at.selectbox(key="opt_model").options, ["Auto", "a", "b"])  # cloud off
            next(g for g in at.button_group if g.key == "opt_cloud").set_value("Allowed").run()
            offered = at.selectbox(key="opt_model").options
            self.assertIn("openrouter/free", offered)
            self.assertNotIn("openrouter/auto", offered)
            at.toggle(key="opt_allow_paid").set_value(True).run()
            self.assertIn("openrouter/auto", at.selectbox(key="opt_model").options)

    def test_an_answer_from_your_documents_is_flagged_in_later_history(self):
        real = self.fake_answer

        def grounded(question, **kwargs):
            result = real(question, **kwargs)
            if question == "what do my notes say?":
                result.update(needs_rag=True, sources=[{"source": "n.md", "text": "note text"}])
            return result

        with mock.patch.object(engine, "answer", grounded), \
                mock.patch("nexus.ui.run_truth"):
            at = self.start()
            self.ask(at, "what do my notes say?")
            self.ask(at, "thanks")
        history = self.calls[-1]["history"]
        self.assertEqual(history[1].get("meta"), {"needs_rag": True})
        self.assertNotIn("meta", history[0])

    def test_the_ui_does_not_start_background_work_under_test(self):
        # tests/isolate.py switches the brain off; a watcher thread left
        # running would open other tests' databases for the rest of the run.
        import threading

        from nexus import config

        self.assertFalse(config.get_settings().brain_enabled)
        at = self.start()
        at.switch_page("app_pages/inbox.py").run()
        self.assertFalse(at.exception, at.exception)
        self.assertNotIn("nexus-brain", [t.name for t in threading.enumerate()])

    def test_deleting_a_chat(self):
        at = self.start()
        self.ask(at, "to be deleted")
        [chat] = store.list_chats()
        at.button(key=f"del_{chat['id']}").click().run()
        at.button(key="confirm_delete_chat").click().run()
        self.assertEqual(store.list_chats(), [])


if __name__ == "__main__":
    unittest.main()
