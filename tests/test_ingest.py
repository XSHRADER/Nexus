import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import isolate  # noqa: F401  (must precede any nexus import)

from nexus import ingest
from nexus.retrieve import Retriever


class IngestTests(unittest.TestCase):
    def setUp(self):
        root = Path(tempfile.mkdtemp(prefix="nexus_ingest_"))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        self.docs = root / "documents"
        self.index = root / "vector_store"
        self.docs.mkdir()
        self.write("a.md", "Alpha notes about the launcher, start.bat.")
        self.write("sub/b.txt", "Beta notes about Chroma and embeddings.")

    def write(self, name, text):
        path = self.docs / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def run_ingest(self, **kwargs):
        lines = []
        report = ingest.run(docs_dir=self.docs, index_dir=self.index, echo=lines.append, **kwargs)
        return report, lines

    def sources(self):
        return sorted(Retriever(db_dir=self.index).stats()["files"])

    def test_first_run_indexes_everything(self):
        report, _ = self.run_ingest()
        self.assertEqual(sorted(report.added), ["a.md", "sub/b.txt"])
        self.assertEqual(self.sources(), ["a.md", "sub/b.txt"])

    def test_second_run_reads_nothing(self):
        self.run_ingest()
        with mock.patch.object(ingest, "chunk_file") as chunk_file:
            report, _ = self.run_ingest()
        chunk_file.assert_not_called()
        self.assertFalse(report.changed)

    def test_only_the_changed_file_is_read(self):
        # Every run used to load and chunk the whole folder, then throw away
        # all but the changed files.
        self.run_ingest()
        self.write("a.md", "Alpha notes, revised.")
        real = ingest.chunk_file
        with mock.patch.object(ingest, "chunk_file", side_effect=real) as chunk_file:
            report, _ = self.run_ingest()
        self.assertEqual([c.args[1] for c in chunk_file.call_args_list], ["a.md"])
        self.assertEqual(report.added, ["a.md"])

    def test_deleted_file_leaves_the_index(self):
        self.run_ingest()
        (self.docs / "a.md").unlink()
        report, _ = self.run_ingest()
        self.assertEqual(report.removed, ["a.md"])
        self.assertEqual(self.sources(), ["sub/b.txt"])

    def test_unreadable_file_is_reported_and_retried(self):
        (self.docs / "broken.pdf").write_bytes(b"not a pdf")
        report, _ = self.run_ingest()
        self.assertIn("broken.pdf", report.failed)
        self.assertEqual(self.sources(), ["a.md", "sub/b.txt"])
        cached = json.loads((self.index / "file_hashes.json").read_text(encoding="utf-8"))
        self.assertNotIn("broken.pdf", cached["files"])
        report, _ = self.run_ingest()
        self.assertIn("broken.pdf", report.failed)  # tried again, not skipped

    def test_office_lock_files_are_ignored(self):
        (self.docs / "~$report.docx").write_bytes(b"lock")
        report, _ = self.run_ingest()
        self.assertEqual(report.failed, {})

    def test_rebuild_under_a_live_retriever(self):
        # The UI keeps a retriever open while the user clicks "Full rebuild".
        self.run_ingest()
        live = Retriever(db_dir=self.index)
        self.write("c.md", "Gamma notes about BM25 keyword search.")
        report, _ = self.run_ingest(force_rebuild=True)
        self.assertTrue(report.rebuilt)
        hits = live.query("BM25 keyword", top_k=1, rerank=False)
        self.assertEqual(hits[0]["meta"]["source"], "c.md")

    def test_changed_settings_force_a_rebuild(self):
        self.run_ingest()
        changed = dict(ingest.index_config(), chunker="older")
        with mock.patch.object(ingest, "index_config", return_value=changed):
            self.run_ingest()
        report, _ = self.run_ingest()
        self.assertTrue(report.rebuilt)
        self.assertEqual(self.sources(), ["a.md", "sub/b.txt"])


if __name__ == "__main__":
    unittest.main()
