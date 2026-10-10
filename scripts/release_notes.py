"""
release_notes.py
Prints the CHANGELOG.md section for one version, after checking that the
version is the same everywhere it is written down. Used by the release
workflow, and by tests/test_version.py so a mismatch fails before tagging.

Usage
    python scripts/release_notes.py v0.4.0     # notes for that tag
    python scripts/release_notes.py            # notes for the current version
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _find(path: str, pattern: str) -> str:
    match = re.search(pattern, (ROOT / path).read_text(encoding="utf-8"), re.M)
    if not match:
        raise SystemExit(f"No version found in {path}")
    return match.group(1)


def versions() -> dict[str, str]:
    """Where the version is written -> what it says there."""
    return {
        "pyproject.toml": _find("pyproject.toml", r'^version\s*=\s*"([^"]+)"'),
        "nexus/__init__.py": _find("nexus/__init__.py", r'^__version__\s*=\s*"([^"]+)"'),
        "CHANGELOG.md": _find("CHANGELOG.md", r"^## \[([^\]]+)\]"),
    }


def notes(version: str) -> str:
    """The changelog section for `version`, without its heading."""
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    match = re.search(rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)", text, re.M | re.S)
    if not match:
        raise SystemExit(f"CHANGELOG.md has no section for {version}")
    return match.group(1).strip() + "\n"


def main(argv: list[str]) -> int:
    found = versions()
    wanted = argv[1].removeprefix("v") if len(argv) > 1 else found["pyproject.toml"]
    wrong = {where: v for where, v in found.items() if v != wanted}
    if wrong:
        raise SystemExit(f"Version {wanted} expected, but found: {wrong}")
    sys.stdout.write(notes(wanted))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
