"""
pc_agent.py
Natural-language front end for PCToolkit.

Turns "sort my downloads folder" into a concrete operation on a concrete
path. Read-only operations (analyze, find duplicates, find large files)
run immediately. Operations that change the disk (organize, delete empty
folders) return a PREVIEW plus a `pending` action. `handle()` itself never
changes the disk; only `apply(pending)` does.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from nexus.pc_tools import PCToolkit

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

LARGE_FILE_MB = 50.0
LIST_LIMIT = 25


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


def _describe_known(path: Path) -> str:
    try:
        return f"~\\{path.relative_to(Path.home())}"
    except ValueError:
        return str(path)


def resolve_target(query: str, base_dir: Path) -> tuple[Path, str]:
    """Return (path, human-readable description of how it was chosen)."""
    quoted = _QUOTED_RE.search(query)
    if quoted:
        candidate = quoted.group(1).strip()
        if (re.match(r"[a-zA-Z]:[\\/]", candidate) or candidate.startswith("~")
                or "/" in candidate or "\\" in candidate):
            return Path(candidate).expanduser(), "path in your message"

    drive = _DRIVE_PATH_RE.search(query)
    if drive:
        return Path(drive.group(0)), "path in your message"

    low = query.lower()
    for keyword, folder in KNOWN_DIRS.items():
        if re.search(rf"\b{keyword}\b", low):
            resolved = _known_dir(folder)
            return resolved, _describe_known(resolved)

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


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _size(mb: float) -> str:
    return f"{mb / 1024:.2f} GB" if mb >= 1024 else f"{mb:.1f} MB"


def _format_plan(moves: list[dict[str, str]], limit: int = 40) -> str:
    by_cat: dict[str, list[str]] = {}
    for mv in moves:
        by_cat.setdefault(mv["category"], []).append(Path(mv["destination"]).name)
    lines = []
    for cat, names in sorted(by_cat.items()):
        lines.append(f"**{cat}/** — {len(names)} file(s)")
        lines.extend(f"- {name}" for name in names[:limit])
        if len(names) > limit:
            lines.append(f"- …and {len(names) - limit} more")
    return "\n".join(lines)


def _format_analysis(info: dict[str, Any], how: str) -> str:
    rows = [f"| {cat} | {n} |" for cat, n in info["category_counts"].items() if n]
    table = "\n".join(["| Type | Files |", "|---|---:|", *rows]) if rows else "_No files._"
    return (f"📂 **`{info['path']}`** ({how}) — {info['total_files']} file(s), "
            f"{_size(info['total_size_mb'])} in total\n\n{table}")


def _format_duplicates(target: Path, groups: list[dict[str, Any]]) -> str:
    wasted = sum(g["size_mb"] * (len(g["paths"]) - 1) for g in groups)
    lines = [f"🔍 **{len(groups)} set(s) of identical files** in `{target}` — "
             f"{_size(wasted)} could be freed by keeping one of each:\n"]
    for g in groups[:LIST_LIMIT]:
        lines.append(f"- **{_size(g['size_mb'])}** × {len(g['paths'])}")
        lines.extend(f"  - `{p}`" for p in g["paths"])
    if len(groups) > LIST_LIMIT:
        lines.append(f"- …and {len(groups) - LIST_LIMIT} more set(s)")
    return "\n".join(lines)


def _format_large(target: Path, files: list[dict[str, Any]]) -> str:
    lines = [f"📊 **Files over {LARGE_FILE_MB:.0f} MB** in `{target}`:\n",
             "| Size | File |", "|---:|---|"]
    lines.extend(f"| {_size(f['size_mb'])} | `{f['path']}` |" for f in files[:LIST_LIMIT])
    if len(files) > LIST_LIMIT:
        lines.append(f"\n…and {len(files) - LIST_LIMIT} more.")
    return "\n".join(lines)


def _read_only(action: str, answer: str) -> dict[str, Any]:
    return {"answer": answer, "action": action, "requires_confirmation": False, "pending": None}


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def handle(query: str, base_dir: Path) -> dict[str, Any]:
    """Route `query` to a PCToolkit operation.

    Never changes the disk: a mutating request comes back as a preview plus a
    `pending` action that only `apply()` executes.
    """
    intent = parse_intent(query)
    target, how = resolve_target(query, base_dir)
    try:
        return _handle(intent, target, how)
    except (ValueError, OSError) as exc:
        return _read_only(intent, f"⚠️ {exc}")


def _handle(intent: str, target: Path, how: str) -> dict[str, Any]:
    if intent == "duplicates":
        groups = _toolkit.find_duplicates(target)
        if not groups:
            return _read_only(intent, f"🔍 No duplicate files in `{target}`.")
        return _read_only(intent, _format_duplicates(target, groups))

    if intent == "large":
        large = _toolkit.find_large_files(target, min_size_mb=LARGE_FILE_MB)
        if not large:
            return _read_only(intent, f"📊 No files over {LARGE_FILE_MB:.0f} MB in `{target}`.")
        return _read_only(intent, _format_large(target, large))

    if intent == "analyze":
        return _read_only(intent, _format_analysis(_toolkit.analyze_directory(target), how))

    if intent == "undo":
        res = _toolkit.undo_last_organize(target)
        note = f"↩️ Restored **{res['restored']}** file(s) to their original places in `{target}`."
        if res["errors"]:
            note += "\n\nNot restored:\n" + "\n".join(f"- {e}" for e in res["errors"])
        return _read_only(intent, note)

    if intent == "empty_dirs":
        preview = _toolkit.delete_empty_dirs(target, dry_run=True)
        if not preview["removed"]:
            return _read_only(intent, f"🧹 No empty folders under `{target}`.")
        listing = "\n".join(f"- `{p}`" for p in preview["removed"][:50])
        more = f"\n- …and {preview['removed_count'] - 50} more" if preview["removed_count"] > 50 else ""
        return {
            "answer": f"🧹 **{preview['removed_count']} empty folder(s)** under `{target}` "
                      f"would be removed:\n\n{listing}{more}\n\n_Confirm to delete them._",
            "action": intent,
            "requires_confirmation": True,
            "pending": {"op": "empty_dirs", "path": str(target)},
        }

    # intent == "organize"
    plan = _toolkit.organize_folder(target, dry_run=True)
    if not plan["planned_moves"]:
        return _read_only(intent, f"🗂️ `{target}` is already sorted — nothing to move.")
    return {
        "answer": f"🗂️ **Plan for `{target}`** ({how}) — {plan['moved_count']} file(s) "
                  f"into category folders:\n\n{_format_plan(plan['planned_moves'])}\n\n"
                  f"_Confirm to move them. This is undoable afterwards._",
        "action": intent,
        "requires_confirmation": True,
        "pending": {"op": "organize", "path": str(target)},
    }


def apply(pending: dict[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
    """Execute an action that a previous `handle` call returned as `pending`."""
    op = pending.get("op")
    target = Path(pending["path"]).resolve()
    if op == "organize":
        res = _toolkit.organize_folder(target, dry_run=False)
        note = f"✅ Moved **{res['moved_count']}** file(s) in `{target}` into category folders."
        if res["error"]:
            note += f"\n\n⚠️ {res['error']}"
        if res["moved_count"]:
            note += f'\n\nTo undo: `undo organize in "{target}"`'
        return _read_only(op, note)
    if op == "empty_dirs":
        res = _toolkit.delete_empty_dirs(target, dry_run=False)
        return _read_only(op, f"🧹 Removed **{res['removed_count']}** empty folder(s) under `{target}`.")
    raise ValueError(f"Unknown pending operation: {op!r}")
