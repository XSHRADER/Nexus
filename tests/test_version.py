import re
import sys
import unittest
from pathlib import Path

import isolate  # noqa: F401  (must precede any nexus import)

import nexus

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import release_notes


class VersionTests(unittest.TestCase):
    def test_version_is_the_same_everywhere(self):
        # pyproject.toml, nexus/__init__.py and the newest changelog entry.
        found = release_notes.versions()
        self.assertEqual(set(found.values()), {nexus.__version__}, found)

    def test_version_is_major_minor_patch(self):
        self.assertRegex(nexus.__version__, r"^\d+\.\d+\.\d+$")

    def test_current_version_has_release_notes(self):
        self.assertTrue(release_notes.notes(nexus.__version__).strip())

    def test_every_changelog_version_is_in_the_roadmap_as_released(self):
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        roadmap = (ROOT / "ROADMAP.md").read_text(encoding="utf-8")
        for version in re.findall(r"^## \[([^\]]+)\]", changelog, re.M):
            self.assertRegex(roadmap, rf"\| \*\*{re.escape(version)}\*\* \| Released \|", version)


if __name__ == "__main__":
    unittest.main()
