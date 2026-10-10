"""Background brain (Phase 6).

Uses a temporary documents folder, the mock Ollama's flashcard and digest
writers as the "local model", and the keyword truth checker, so it is
deterministic and needs no models.
"""

import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from dataclasses import replace
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demos"))

import fakes  # noqa: E402
import isolate  # noqa: F401  (must precede any nexus import)
import mock_ollama  # noqa: E402

from nexus import (
    brain,  # noqa: E402
    cloud,  # noqa: E402
    cloud_client,  # noqa: E402
    config,  # noqa: E402
    engine,  # noqa: E402
    feedback,  # noqa: E402
    ollama,  # noqa: E402
    store,  # noqa: E402
    truth_check,  # noqa: E402
)

NOTES = ("# Vector stores\n\nThe project uses Chroma for persistent vector storage on disk. "
         "Ollama provides local answer generation with the llama3.1:8b model for every reply.\n\n"
         "Chunks are packed to 240 tokens with 48 tokens of overlap so nothing is truncated "
         "before it reaches the embedding model.\n")


def fake_generate(messages):
    return mock_ollama.reply_for(messages, "llama3.1:8b"), "llama3.1:8b"


class BrainTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.docs = root / "documents"
        self.docs.mkdir()
        self.extra = root / "notes"
        self.extra.mkdir()
        self.store = fakes.use_db(root / "t.db")
        self.indexed = 0
        self.settings = config.Settings(digest_days=7.0)
        self.patches = [mock.patch.object(brain, "DOCS_DIR", self.docs),
                        mock.patch.object(brain, "get_settings", side_effect=lambda: self.settings),
                        mock.patch.object(truth_check, "get_settings", side_effect=lambda: self.settings)]
        for p in self.patches:
            p.start()
        self.brain = brain.Brain(store=self.store, generate=fake_generate, index=self.count_index,
                                 folders=lambda: [self.docs, self.extra],
                                 scorer=truth_check.KeywordScorer())
        self.brain.tick()  # first look: remembers what exists, posts nothing

    def tearDown(self):
        for p in self.patches:
            p.stop()
        fakes.restore_db()
        self.tmp.cleanup()

    def count_index(self):
        self.indexed += 1


class WatchTests(BrainTestCase):
    def test_first_scan_is_quiet(self):
        (self.docs / "old.md").write_text("Existing note that was there before.")
        fresh = brain.Brain(store=store.ChatStore(Path(self.tmp.name) / "other.db"),
                            generate=fake_generate, index=self.count_index,
                            folders=lambda: [self.docs], scorer=truth_check.KeywordScorer())
        report = fresh.tick()
        self.assertTrue(report["first_run"])
        self.assertEqual(fresh.inbox(), [])
        self.assertEqual(self.indexed, 0)

    def test_new_note_is_indexed_and_becomes_verified_flashcards(self):
        (self.docs / "stack.md").write_text(NOTES)
        report = self.brain.tick()
        self.assertEqual(report["changes"]["added"], [str((self.docs / "stack.md").resolve())])
        self.assertEqual(self.indexed, 1)
        kinds = [i["kind"] for i in self.brain.inbox()]
        self.assertEqual(sorted(kinds), ["cards", "indexed"])
        cards = self.brain.due_cards()
        self.assertGreaterEqual(len(cards), 2)
        self.assertTrue(all(c["source"].endswith("stack.md") for c in cards))
        self.assertTrue(any(c["check_label"] == "supported" for c in cards))
        # The mock writes one card with a wrong number; the truth check drops it.
        self.assertFalse(any("(check)" in c["question"] for c in cards))
        cards_item = next(i for i in self.brain.inbox() if i["kind"] == "cards")
        self.assertGreaterEqual(cards_item["data"]["discarded"], 1)
        self.assertEqual(self.brain.tick()["changes"],
                         {"added": [], "changed": [], "removed": []})  # nothing new

    def test_change_and_removal_reindex(self):
        note = self.docs / "a.md"
        note.write_text(NOTES)
        self.brain.tick()
        time.sleep(0.01)
        note.write_text(NOTES + "\nA new paragraph about BM25 keyword search is here.\n")
        os.utime(note, (time.time() + 5, time.time() + 5))
        self.assertTrue(self.brain.tick()["changes"]["changed"])
        note.unlink()
        self.assertTrue(self.brain.tick()["changes"]["removed"])
        self.assertEqual(self.indexed, 3)
        with self.store._connect() as db:
            changes = [r["change"] for r in db.execute("SELECT change FROM file_events ORDER BY id")]
        self.assertEqual(changes[-3:], ["added", "changed", "removed"])

    def test_extra_folders_feed_cards_but_are_not_indexed(self):
        (self.extra / "lecture.md").write_text(NOTES)
        self.brain.tick()
        self.assertEqual(self.indexed, 0)
        self.assertTrue(self.brain.due_cards())

    def test_index_failure_is_reported_and_cards_still_made(self):
        self.brain._index = lambda: (_ for _ in ()).throw(RuntimeError("chromadb missing"))
        (self.docs / "b.md").write_text(NOTES)
        self.brain.tick()
        errors = [i for i in self.brain.inbox() if i["kind"] == "error"]
        self.assertIn("chromadb missing", errors[0]["body"])
        self.assertTrue(self.brain.due_cards())

    def test_study_can_be_switched_off(self):
        self.settings = replace(self.settings, study_enabled=False)
        (self.docs / "c.md").write_text(NOTES)
        self.brain.tick()
        self.assertEqual(self.brain.due_cards(), [])

    def test_study_all_uses_existing_notes(self):
        (self.docs / "d.md").write_text(NOTES)
        self.brain._remember(brain.scan([self.docs]), {"added": [], "changed": [], "removed": []})
        self.assertGreater(self.brain.study_all(), 0)


class StudyTests(BrainTestCase):
    def test_leitner_schedule(self):
        now = 1000.0
        self.assertEqual(brain.next_due(1, True, now), (2, now + 1 * brain.DAY))
        self.assertEqual(brain.next_due(4, True, now), (5, now + 14 * brain.DAY))
        self.assertEqual(brain.next_due(5, True, now), (5, now + 14 * brain.DAY))
        self.assertEqual(brain.next_due(4, False, now), (1, now + brain.RETRY_SECONDS))

    def test_review_moves_cards_out_of_the_due_list(self):
        self.brain._save_card("x.md", {"q": "What is X here?", "a": "X is Y."}, "X is Y.", "supported")
        (card,) = self.brain.due_cards()
        self.brain.review(card["id"], knew=True)
        self.assertEqual(self.brain.due_cards(), [])
        self.assertEqual(len(self.brain.due_cards(now=time.time() + 2 * brain.DAY)), 1)
        self.assertEqual(self.brain.study_stats()["total"], 1)
        with self.assertRaises(ValueError):
            self.brain.review(999, True)

    def test_parse_cards(self):
        text = 'Sure! [{"q": "What stores vectors?", "a": "Chroma."}, {"question": "Q?", "answer": "x"}, 5]'
        self.assertEqual(brain.parse_cards(text), [{"q": "What stores vectors?", "a": "Chroma."}])
        self.assertEqual(brain.parse_cards("no cards"), [])


class DigestTests(BrainTestCase):
    def test_digest_is_weekly(self):
        now = time.time()
        self.assertFalse(self.brain.digest_due(now))  # first call starts the clock
        self.assertFalse(self.brain.digest_due(now + 3 * brain.DAY))
        self.assertTrue(self.brain.digest_due(now + 8 * brain.DAY))
        self.settings = replace(self.settings, digest_days=0)
        self.assertFalse(self.brain.digest_due(now + 30 * brain.DAY))

    def test_digest_covers_files_questions_feedback_and_study(self):
        self.brain.digest_due()
        (self.docs / "stack.md").write_text(NOTES)
        self.brain.tick()
        cid = self.store.create_chat("q")
        self.store.add_message(cid, "user", "Explain vector databases please")
        mid = self.store.add_message(cid, "assistant", "...", {"task": "general", "model": "m"})
        feedback.rate(mid, 1)
        item_id = self.brain.make_digest()
        (item,) = [i for i in self.brain.inbox() if i["id"] == item_id]
        body = item["body"]
        self.assertIn("New: documents/stack.md".replace("documents/", ""), body.replace("documents/", ""))
        self.assertIn("1 question(s)", body)
        self.assertIn("1 👍", body)
        self.assertIn("stack.md: The project uses Chroma", body)  # local model summary
        self.assertIn("summarised on this PC by llama3.1:8b", body)
        self.assertIn("flashcard(s)", body)
        self.assertFalse(self.brain.digest_due())  # just written

    def test_digest_without_a_model_still_reports(self):
        self.brain._generate = lambda m: (_ for _ in ()).throw(RuntimeError("Ollama down"))
        (self.docs / "e.md").write_text(NOTES)
        self.brain.tick()
        item_id = self.brain.make_digest()
        body = next(i for i in self.brain.inbox() if i["id"] == item_id)["body"]
        self.assertIn("no local model could summarise", body)


class LocalOnlyTests(unittest.TestCase):
    def test_local_generate_ignores_the_cloud_switch(self):
        server, url = mock_ollama.start_in_thread()
        settings = config.Settings(cloud_mode="allowed", allow_docs_to_cloud=True, allow_paid=True)
        try:
            with mock.patch.object(config, "OLLAMA_URL", url), \
                 mock.patch.object(cloud, "get_settings", return_value=settings), \
                 mock.patch.dict(os.environ, {"GEMINI_API_KEY": "k"}), \
                 mock.patch.object(ollama, "installed_models", return_value=["llama3.1:8b"]), \
                 mock.patch.object(ollama, "loaded_models", return_value=[]), \
                 mock.patch.object(cloud_client, "chat", side_effect=AssertionError("cloud used")):
                text, model = engine.local_generate([{"role": "user", "content": "hello"}])
        finally:
            server.shutdown()
        self.assertEqual(model, "llama3.1:8b")
        self.assertTrue(text)


class ServerAndThreadTests(BrainTestCase):
    def test_endpoints(self):
        from nexus import server

        brain._brain = self.brain
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"

        def call(path, payload=None):
            data = json.dumps(payload).encode() if payload is not None else None
            req = urllib.request.Request(base + path, data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req) as res:
                return json.loads(res.read())

        try:
            (self.docs / "f.md").write_text(NOTES)
            ran = call("/api/brain/run", {"action": "scan"})
            self.assertEqual(ran["unread"], 2)
            inbox = call("/api/inbox")
            self.assertEqual(len(inbox["items"]), 2)
            study = call("/api/study")
            self.assertTrue(study["cards"])
            out = call("/api/study/answer", {"card_id": study["cards"][0]["id"], "knew": True})
            self.assertEqual(out["box"], 2)
            self.assertEqual(call("/api/inbox/read", {})["unread"], 0)
            self.assertIn("brain", call("/api/status"))
        finally:
            httpd.shutdown()
            brain._brain = None

    def test_background_thread_notices_a_new_file(self):
        self.settings = replace(self.settings, brain_scan_seconds=0.05)
        self.brain.start()
        try:
            (self.docs / "g.md").write_text(NOTES)
            deadline = time.time() + 5
            while time.time() < deadline and not self.brain.inbox():
                time.sleep(0.05)
        finally:
            self.brain.stop()
        self.assertTrue(self.brain.inbox())


if __name__ == "__main__":
    unittest.main()
