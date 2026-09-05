import tempfile
import unittest
from pathlib import Path

from pc_agent import apply, handle, parse_intent, resolve_target
from pc_tools import MANIFEST_NAME, PCToolkit


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
            preview = handle(f'sort "{root}"', root, confirm=False)
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

    def test_safety_refuses_home_dir(self):
        with self.assertRaises(ValueError):
            PCToolkit().organize_folder(Path.home(), dry_run=True)


if __name__ == "__main__":
    unittest.main()
