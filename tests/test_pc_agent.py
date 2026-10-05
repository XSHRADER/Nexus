import tempfile
import unittest
from pathlib import Path

from nexus.pc_agent import apply, handle, parse_intent, resolve_target
from nexus.pc_tools import MANIFEST_NAME, PCToolkit


class ParseIntentTests(unittest.TestCase):
    def test_sort_is_organize(self):
        self.assertEqual(parse_intent("sort my downloads folder"), "organize")

    def test_tidy_is_organize(self):
        self.assertEqual(parse_intent("tidy up the desktop please"), "organize")

    def test_clean_up_is_organize(self):
        self.assertEqual(parse_intent("can you clean up this folder"), "organize")

    def test_undo(self):
        self.assertEqual(parse_intent("undo that last organize"), "undo")

    def test_empty_dirs(self):
        self.assertEqual(parse_intent("remove empty folders here"), "empty_dirs")

    def test_duplicates(self):
        self.assertEqual(parse_intent("find duplicate files"), "duplicates")

    def test_large(self):
        self.assertEqual(parse_intent("show the biggest files taking up space"), "large")

    def test_default_is_analyze(self):
        self.assertEqual(parse_intent("what is in this directory"), "analyze")


class ResolveTargetTests(unittest.TestCase):
    def test_known_folder(self):
        path, _ = resolve_target("organize my downloads", Path.cwd())
        self.assertEqual(path, Path.home() / "Downloads")

    def test_quoted_path_wins(self):
        path, _ = resolve_target('sort "C:/Temp/demo"', Path.cwd())
        self.assertEqual(str(path).replace("\\", "/"), "C:/Temp/demo")

    def test_default_to_base_dir(self):
        base = Path.cwd()
        path, how = resolve_target("analyze disk usage", base)
        self.assertEqual(path, base)
        self.assertIn("default", how)


class OrganizeFlowTests(unittest.TestCase):
    def test_preview_then_apply_then_undo(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "a.txt").write_text("x", encoding="utf-8")
            (root / "b.png").write_bytes(b"x")
            (root / ".keep").write_text("x", encoding="utf-8")  # hidden -> untouched

            # preview
            preview = handle(f'sort "{root}"', root)
            self.assertTrue(preview["requires_confirmation"])
            self.assertEqual(preview["pending"]["op"], "organize")
            self.assertTrue((root / "a.txt").exists())  # nothing moved yet

            # apply
            done = apply(preview["pending"])
            self.assertFalse(done["requires_confirmation"])
            self.assertTrue((root / "Documents" / "a.txt").exists())
            self.assertTrue((root / "Images" / "b.png").exists())
            self.assertTrue((root / ".keep").exists())
            self.assertTrue((root / MANIFEST_NAME).exists())

            # undo
            res = PCToolkit().undo_last_organize(root)
            self.assertEqual(res["restored"], 2)
            self.assertTrue((root / "a.txt").exists())
            self.assertFalse((root / MANIFEST_NAME).exists())

    def test_handle_never_changes_the_disk(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "empty").mkdir()
            (root / "a.txt").write_text("x", encoding="utf-8")
            for query in (f'remove empty folders in "{root}"', f'sort "{root}"'):
                result = handle(query, root)
                self.assertTrue(result["requires_confirmation"], query)
                with self.assertRaises(TypeError):
                    handle(query, root, confirm=True)
            self.assertTrue((root / "empty").is_dir())
            self.assertTrue((root / "a.txt").exists())

    def test_safety_refuses_home_dir(self):
        with self.assertRaises(ValueError):
            PCToolkit().organize_folder(Path.home(), dry_run=True)

    def test_missing_folder_is_an_answer_not_a_crash(self):
        result = handle('analyze "C:/definitely/not/here"', Path.cwd())
        self.assertIn("Not an existing folder", result["answer"])


class ToolkitSafetyTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.kit = PCToolkit()

    def test_empty_dir_cleanup_never_enters_ignored_folders(self):
        # The bottom-up walk ignored its own skip list, so empty folders
        # inside .git and virtual envs were deleted.
        (self.root / ".git" / "refs" / "tags").mkdir(parents=True)
        (self.root / "node_modules" / "pkg").mkdir(parents=True)
        (self.root / "really_empty").mkdir()
        res = self.kit.delete_empty_dirs(self.root, dry_run=False)
        self.assertEqual(res["removed"], [str(self.root / "really_empty")])
        self.assertTrue((self.root / ".git" / "refs" / "tags").is_dir())
        self.assertTrue((self.root / "node_modules" / "pkg").is_dir())

    def test_empty_dir_preview_matches_the_real_run(self):
        # A folder holding only empty folders was missing from the preview
        # (its child still existed during a dry run) but removed for real.
        (self.root / "outer" / "inner" / "deepest").mkdir(parents=True)
        (self.root / "kept").mkdir()
        (self.root / "kept" / "file.txt").write_text("x", encoding="utf-8")
        preview = self.kit.delete_empty_dirs(self.root, dry_run=True)
        real = self.kit.delete_empty_dirs(self.root, dry_run=False)
        self.assertEqual(preview["removed_count"], 3)
        self.assertEqual(sorted(preview["removed"]), sorted(real["removed"]))
        self.assertFalse((self.root / "outer").exists())
        self.assertTrue((self.root / "kept").is_dir())

    def test_undo_never_overwrites_a_new_file(self):
        (self.root / "a.txt").write_text("original", encoding="utf-8")
        self.kit.organize_folder(self.root, dry_run=False)
        (self.root / "a.txt").write_text("new file, same name", encoding="utf-8")
        res = self.kit.undo_last_organize(self.root)
        self.assertEqual(res["restored"], 0)
        self.assertEqual((self.root / "a.txt").read_text(encoding="utf-8"), "new file, same name")
        self.assertEqual((self.root / "Documents" / "a.txt").read_text(encoding="utf-8"), "original")
        self.assertTrue((self.root / MANIFEST_NAME).exists())  # still undoable later

    def test_each_organize_run_is_undone_separately(self):
        # The manifest used to be overwritten, so the first run became
        # impossible to undo once a second one happened.
        (self.root / "a.txt").write_text("x", encoding="utf-8")
        self.kit.organize_folder(self.root, dry_run=False)
        (self.root / "b.png").write_bytes(b"x")
        self.kit.organize_folder(self.root, dry_run=False)
        self.assertEqual(self.kit.undo_last_organize(self.root)["restored"], 1)
        self.assertTrue((self.root / "b.png").exists())
        self.assertEqual(self.kit.undo_last_organize(self.root)["restored"], 1)
        self.assertTrue((self.root / "a.txt").exists())
        self.assertFalse((self.root / MANIFEST_NAME).exists())

    def test_a_failure_part_way_keeps_the_finished_moves_undoable(self):
        for name in ("a.txt", "b.txt", "c.txt"):
            (self.root / name).write_text(name, encoding="utf-8")
        import shutil as _shutil
        real_move = _shutil.move
        calls = []

        def flaky(src, dst):
            calls.append(src)
            if len(calls) == 2:
                raise PermissionError("file is open in another program")
            return real_move(src, dst)

        from unittest import mock
        with mock.patch("nexus.pc_tools.shutil.move", flaky):
            res = self.kit.organize_folder(self.root, dry_run=False)
        self.assertEqual(res["moved_count"], 1)
        self.assertIn("Stopped after 1 of 3", res["error"])
        self.assertEqual(self.kit.undo_last_organize(self.root)["restored"], 1)
        self.assertTrue((self.root / "a.txt").exists())

    def test_duplicates_need_identical_content_not_just_size(self):
        (self.root / "one.bin").write_bytes(b"A" * 100_000)
        (self.root / "two.bin").write_bytes(b"A" * 100_000)
        (self.root / "same_size.bin").write_bytes(b"A" * 99_999 + b"B")
        groups = self.kit.find_duplicates(self.root)
        self.assertEqual(len(groups), 1)
        self.assertEqual(sorted(Path(p).name for p in groups[0]["paths"]), ["one.bin", "two.bin"])


if __name__ == "__main__":
    unittest.main()
