"""
pc_agent.py
Natural-language front end for PCToolkit.

Turns "sort my downloads folder" into a concrete operation on a concrete
path. Read-only operations (analyze, find duplicates, find large files)
run immediately. Operations that change the disk (organize, delete empty
folders) return a PREVIEW plus a `pending` action; they only execute when
`handle(..., confirm=True)` or `apply(pending)` is called.
"""

import json
import os
import re
from pathlib import Path
from typing import Any

from pc_tools import PCToolkit

_toolkit = PCToolkit()

# keyword in the query  ->  standard user folder name
KNOWN_DIRS = {
    "downloads": "Downloads",
    "download": "Downloads",
    "desktop": "Desktop",
    "documents": "Documents",
    "docs": "Documents",
    "pictures": "Pictures",
    "photos": "Pictures",
    "music": "Music",
    "videos": "Videos",
    "movies": "Videos",
}

_DRIVE_PATH_RE = re.compile(r"[a-zA-Z]:[\\/][^\s\"'<>|?*\n]*")
_QUOTED_RE = re.compile(r"[\"'`]([^\"'`\n]+)[\"'`]")


def _known_dir(folder: str) -> Path:
    """Locate a shell folder like Documents.

    On Windows with OneDrive backup turned on, Documents/Desktop/Pictures live
    under `~/OneDrive/` and `~/Documents` doesn't exist at all — so prefer
    whichever actually exists, checking OneDrive before falling back.
    """
    home = Path.home()
    candidates = [home / folder]
    onedrive = os.environ.get("OneDrive") or os.environ.get("OneDriveConsumer")
    if onedrive:
        candidates.insert(0, Path(onedrive) / folder)
    candidates.append(home / "OneDrive" / folder)

    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return home / folder  # let the toolkit report the missing path


def _describe_known(path: Path, folder: str) -> str:
    try:
        return f"~\\{path.relative_to(Path.home())}"
    except ValueError:
        return str(path)


def resolve_target(query: str, base_dir: Path) -> tuple[Path, str]:
    """Return (path, human-readable description of how it was chosen)."""
    quoted = _QUOTED_RE.search(query)
    if quoted:
        candidate = quoted.group(1).strip()
        if re.match(r"[a-zA-Z]:[\\/]", candidate) or candidate.startswith("~") or "/" in candidate or "\\" in candidate:
            return Path(candidate).expanduser(), "path in your message"

    drive = _DRIVE_PATH_RE.search(query)
    if drive:
        return Path(drive.group(0)), "path in your message"

    low = query.lower()
    for keyword, folder in KNOWN_DIRS.items():
        if re.search(rf"\b{keyword}\b", low):
            resolved = _known_dir(folder)
            return resolved, _describe_known(resolved, folder)

    if "my home" in low or re.search(r"\bhome (folder|directory)\b", low):
        return Path.home(), "home directory"
    if re.search(r"\b(this|current|the) (folder|directory|repo|project)\b", low):
        return base_dir, "project folder"

    return base_dir, "project folder (default)"


def parse_intent(query: str) -> str:
    low = query.lower()
    if re.search(r"\b(undo|revert|restore|put .*back)\b", low):
        return "undo"
    if re.search(r"\b(sort|organi[sz]e|tidy|arrange|categori[sz]e|declutter)\b", low) or "clean up" in low:
        return "organize"
    if "remove empty" in low or "delete empty" in low or re.search(r"\bempty (folder|folders|dir|directories)\b", low):
        return "empty_dirs"
    if "duplicate" in low:
        return "duplicates"
    if re.search(r"\b(large|big|huge|biggest|largest) (file|files)\b", low) or "taking up space" in low:
        return "large"
    return "analyze"


def _format_plan(moves: list[dict[str, str]], limit: int = 40) -> str:
    by_cat: dict[str, list[str]] = {}
    for mv in moves:
        by_cat.setdefault(mv["category"], []).append(Path(mv["source"]).name)
    lines = []
    for cat, names in sorted(by_cat.items()):
        lines.append(f"**{cat}/** — {len(names)} file(s)")
        for name in names[:limit]:
            lines.append(f"- {name}")
        if len(names) > limit:
            lines.append(f"- …and {len(names) - limit} more")
    return "\n".join(lines)


def _read_only(action: str, answer: str) -> dict[str, Any]:
    return {"answer": answer, "action": action, "requires_confirmation": False, "pending": None}


def handle(query: str, base_dir: Path, confirm: bool = False) -> dict[str, Any]:
    """Route `query` to a PCToolkit operation. See module docstring for the confirm flow."""
    intent = parse_intent(query)
    target, how = resolve_target(query, base_dir)

    if intent == "duplicates":
        dups = _toolkit.find_duplicates(target)
        if not dups:
            return _read_only("duplicates", f"🔍 No duplicate files found in `{target}`.")
        return _read_only(
            "duplicates",
            f"🔍 Found **{len(dups)}** duplicate group(s) in `{target}`:\n\n"
            f"```json\n{json.dumps(dups, indent=2)}\n```",
        )

    if intent == "large":
        large = _toolkit.find_large_files(target, min_size_mb=50.0)
        if not large:
            return _read_only("large", f"📊 No files over 50 MB in `{target}`.")
        return _read_only(
            "large",
            f"📊 Largest files in `{target}`:\n\n```json\n{json.dumps(large[:25], indent=2)}\n```",
        )

    if intent == "analyze":
        info = _toolkit.analyze_directory(target)
        return _read_only(
            "analyze",
            f"📂 **Analysis of `{target}`** ({how}):\n\n```json\n{json.dumps(info, indent=2)}\n```",
        )

    if intent == "undo":
        try:
            res = _toolkit.undo_last_organize(target)
        except ValueError as exc:
            return _read_only("undo", f"↩️ {exc}")
        note = f"↩️ Restored **{res['restored']}** file(s) to their original spots in `{target}`."
        if res["errors"]:
            note += "\n\nProblems:\n" + "\n".join(f"- {e}" for e in res["errors"])
        return _read_only("undo", note)

    if intent == "empty_dirs":
        preview = _toolkit.delete_empty_dirs(target, dry_run=True)
        if not preview["removed"]:
            return _read_only("empty_dirs", f"🧹 No empty folders under `{target}`.")
        if not confirm:
            listing = "\n".join(f"- {p}" for p in preview["removed"][:50])
            return {
                "answer": f"🧹 **{preview['removed_count']} empty folder(s)** under `{target}` "
                          f"would be removed:\n\n{listing}\n\n_Confirm to delete them._",
                "action": "empty_dirs",
                "requires_confirmation": True,
                "pending": {"op": "empty_dirs", "path": str(target)},
            }
        res = _toolkit.delete_empty_dirs(target, dry_run=False)
        return _read_only("empty_dirs", f"🧹 Removed **{res['removed_count']}** empty folder(s) under `{target}`.")

    # intent == "organize"
    if not confirm:
        plan = _toolkit.organize_folder(target, dry_run=True)
        if not plan["planned_moves"]:
            return _read_only("organize", f"🗂️ `{target}` is already sorted — nothing to move.")
        return {
            "answer": f"🗂️ **Plan for `{target}`** ({how}) — {plan['moved_count']} file(s) "
                      f"into category folders:\n\n{_format_plan(plan['planned_moves'])}\n\n"
                      f"_Confirm to move them. This is undoable afterwards._",
            "action": "organize",
            "requires_confirmation": True,
            "pending": {"op": "organize", "path": str(target)},
        }
    return _apply_organize(target)


def _apply_organize(target: Path) -> dict[str, Any]:
    res = _toolkit.organize_folder(target, dry_run=False)
    tail = f'\n\nTo undo: `undo organize in "{target}"`' if res["moved_count"] else ""
    return _read_only(
        "organize",
        f"✅ Moved **{res['moved_count']}** file(s) in `{target}` into category folders.{tail}",
    )


def apply(pending: dict[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
    """Execute an action that a previous `handle` call returned as `pending`."""
    op = pending.get("op")
    target = Path(pending["path"]).resolve()
    if op == "organize":
        return _apply_organize(target)
    if op == "empty_dirs":
        res = _toolkit.delete_empty_dirs(target, dry_run=False)
        return _read_only("empty_dirs", f"🧹 Removed **{res['removed_count']}** empty folder(s) under `{target}`.")
    raise ValueError(f"Unknown pending operation: {op!r}")
