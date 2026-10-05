"""
pc_tools.py
Folder analysis and tidying for NEXUS: usage summaries, duplicate and
large-file scans, sorting loose files into category folders (undoable), and
removing empty folders.

Everything that changes the disk refuses drive roots, the home directory and
system folders, never overwrites a file, and supports a dry run whose preview
matches exactly what the real run does.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

# Directories that are noise for a scan and must never be modified:
# virtual envs, VCS internals, caches, build output, and NEXUS's own index.
IGNORE_DIRS = {
    "nexus-env", ".venv", "venv", "env",
    "__pycache__", ".git", ".svn", ".hg",
    ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "node_modules", ".idea", ".vscode",
    "vector_store",
}

# Written into a folder when it is organized, so the move is reversible.
MANIFEST_NAME = ".nexus_organize_manifest.json"

# Same-size files are first compared on their opening bytes; only files that
# still match are hashed in full. Large media rarely share a prefix, so most
# candidates are ruled out without reading them through.
PREFIX_BYTES = 64 * 1024


def _walk(target: Path):
    """os.walk that prunes IGNORE_DIRS in place."""
    for root, dirs, files in os.walk(target):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        yield root, dirs, files


def _require_dir(target_dir: str | Path) -> Path:
    target = Path(target_dir).resolve()
    if not target.is_dir():
        raise ValueError(f"Not an existing folder: {target}")
    return target


def _assert_safe_target(target: Path) -> None:
    """Refuse to mutate drive roots, the home directory itself, or system paths."""
    target = target.resolve()
    if target.parent == target:
        raise ValueError(f"Refusing to modify a drive root: {target}")
    if target == Path.home().resolve():
        raise ValueError(
            "Refusing to reorganize your home directory directly. "
            "Point me at a subfolder such as Downloads or Desktop."
        )
    dangerous = [
        Path(os.environ.get("SystemRoot", r"C:\Windows")),
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")),
    ]
    for danger in dangerous:
        try:
            danger = danger.resolve()
        except OSError:
            continue
        if target == danger or danger in target.parents:
            raise ValueError(f"Refusing to modify a system directory: {target}")


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    """Organize runs, oldest first. Older NEXUS wrote a single run."""
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if "runs" in data:
        return list(data["runs"])
    return [{"moves": data.get("moves", [])}] if data.get("moves") else []


def _write_manifest(path: Path, target: Path, runs: list[dict[str, Any]]) -> None:
    if not runs:
        path.unlink(missing_ok=True)
        return
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"target": str(target), "runs": runs}, indent=2), encoding="utf-8")
    tmp.replace(path)


class PCToolkit:
    CATEGORY_MAP = {
        "Documents": {".pdf", ".docx", ".doc", ".txt", ".xlsx", ".pptx", ".csv", ".md", ".epub"},
        "Images": {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp", ".ico"},
        "Videos": {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv"},
        "Audio": {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a"},
        "Archives": {".zip", ".rar", ".7z", ".tar", ".gz", ".iso"},
        "Code": {".py", ".js", ".html", ".css", ".json", ".cpp", ".c", ".java", ".ts", ".sh", ".bat"},
        "Executables": {".exe", ".msi", ".apk"},
    }
    CATEGORIES = (*CATEGORY_MAP, "Others")

    @staticmethod
    def file_hash(filepath: str | Path, block_size: int = 65536, limit: int | None = None) -> str:
        """SHA-256 of the file, or of its first `limit` bytes."""
        hasher = hashlib.sha256()
        remaining = limit
        with open(filepath, "rb") as f:
            while remaining is None or remaining > 0:
                size = block_size if remaining is None else min(block_size, remaining)
                block = f.read(size)
                if not block:
                    break
                hasher.update(block)
                if remaining is not None:
                    remaining -= len(block)
        return hasher.hexdigest()

    def _category_for(self, ext: str) -> str:
        for cat, exts in self.CATEGORY_MAP.items():
            if ext in exts:
                return cat
        return "Others"

    # -- read-only scans ------------------------------------------------------

    def analyze_directory(self, target_dir: str | Path) -> dict[str, Any]:
        target = _require_dir(target_dir)
        total_files = 0
        total_size = 0
        categories: dict[str, int] = dict.fromkeys(self.CATEGORIES, 0)
        for root, _, files in _walk(target):
            for fname in files:
                fpath = Path(root) / fname
                try:
                    total_size += fpath.stat().st_size
                except OSError:
                    continue
                total_files += 1
                categories[self._category_for(fpath.suffix.lower())] += 1
        return {
            "path": str(target),
            "total_files": total_files,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "category_counts": categories,
        }

    def _group(self, paths: list[Path], **hash_kwargs) -> list[list[Path]]:
        groups: dict[str, list[Path]] = {}
        for path in paths:
            try:
                groups.setdefault(self.file_hash(path, **hash_kwargs), []).append(path)
            except OSError:
                continue
        return [g for g in groups.values() if len(g) > 1]

    def find_duplicates(self, target_dir: str | Path) -> list[dict[str, Any]]:
        target = _require_dir(target_dir)
        by_size: dict[int, list[Path]] = {}
        for root, _, files in _walk(target):
            for fname in files:
                fpath = Path(root) / fname
                try:
                    size = fpath.stat().st_size
                except OSError:
                    continue
                if size > 0:
                    by_size.setdefault(size, []).append(fpath)

        duplicates = []
        for size, paths in sorted(by_size.items(), reverse=True):
            if len(paths) < 2:
                continue
            candidates = self._group(paths, limit=PREFIX_BYTES) if size > PREFIX_BYTES else [paths]
            for group in candidates:
                for same in self._group(group):
                    duplicates.append({
                        "hash": self.file_hash(same[0])[:12],
                        "size_mb": round(size / (1024 * 1024), 2),
                        "paths": [str(p) for p in same],
                    })
        return duplicates

    def find_large_files(self, target_dir: str | Path, min_size_mb: float = 50.0) -> list[dict[str, Any]]:
        target = _require_dir(target_dir)
        threshold = min_size_mb * 1024 * 1024
        large_files = []
        for root, _, files in _walk(target):
            for fname in files:
                fpath = Path(root) / fname
                try:
                    size = fpath.stat().st_size
                except OSError:
                    continue
                if size >= threshold:
                    large_files.append({"path": str(fpath), "size_mb": round(size / (1024 * 1024), 2)})
        large_files.sort(key=lambda x: x["size_mb"], reverse=True)
        return large_files

    # -- changes ---------------------------------------------------------------

    def organize_folder(self, target_dir: str | Path, dry_run: bool = True) -> dict[str, Any]:
        """Move loose files in `target_dir` into category subfolders.

        Only the top level is touched. Hidden/dotfiles are left alone. A real
        run records every move in a manifest -- including the moves made
        before a failure part-way through -- so `undo_last_organize` can put
        them back. Each run is undone separately, newest first.
        """
        target = _require_dir(target_dir)
        _assert_safe_target(target)

        moves: list[dict[str, str]] = []
        taken: set[Path] = set()  # destinations already claimed by this run
        for fname in sorted(os.listdir(target)):
            fpath = target / fname
            if fpath.is_dir() or fname.startswith("."):
                continue  # leave folders and hidden / config files where they are
            category = self._category_for(fpath.suffix.lower())
            dest_folder = target / category
            dest_file = dest_folder / fname
            n = 1
            while dest_file.exists() or dest_file in taken:  # never overwrite
                dest_file = dest_folder / f"{fpath.stem} ({n}){fpath.suffix}"
                n += 1
            taken.add(dest_file)
            moves.append({"source": str(fpath), "destination": str(dest_file), "category": category})

        manifest_path = target / MANIFEST_NAME
        done: list[dict[str, str]] = []
        error: Exception | None = None
        if not dry_run and moves:
            try:
                for mv in moves:
                    Path(mv["destination"]).parent.mkdir(exist_ok=True)
                    shutil.move(mv["source"], mv["destination"])
                    done.append(mv)
            except OSError as exc:
                error = exc
            finally:
                if done:
                    _write_manifest(manifest_path, target,
                                    _read_manifest(manifest_path) + [{"moves": done}])

        result = {
            "target": str(target),
            "dry_run": dry_run,
            "moved_count": len(moves) if dry_run else len(done),
            "planned_moves": moves,
            "manifest": str(manifest_path) if done else None,
            "error": None,
        }
        if error is not None:
            result["error"] = (f"Stopped after {len(done)} of {len(moves)} moves: {error}. "
                               "The moves made so far can be undone.")
        return result

    def undo_last_organize(self, target_dir: str | Path) -> dict[str, Any]:
        """Reverse the most recent `organize_folder` run using its manifest.

        A file is only moved back if nothing has since taken its old name;
        anything that can't be restored stays in the manifest so a retry is
        possible.
        """
        target = Path(target_dir).resolve()
        manifest_path = target / MANIFEST_NAME
        runs = _read_manifest(manifest_path)
        if not runs:
            raise ValueError(f"No organize manifest in {target} — nothing to undo.")

        restored, errors, left = 0, [], []
        for mv in reversed(runs[-1]["moves"]):
            now_at, back_to = Path(mv["destination"]), Path(mv["source"])
            if not now_at.exists():
                errors.append(f"{now_at.name}: no longer in {now_at.parent.name}/")
                continue
            if back_to.exists():
                errors.append(f"{back_to.name}: a different file now has that name; left in place")
                left.append(mv)
                continue
            try:
                back_to.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(now_at), str(back_to))
                restored += 1
            except OSError as exc:
                errors.append(f"{now_at.name}: {exc}")
                left.append(mv)

        # Drop category folders that are now empty.
        for cat in self.CATEGORIES:
            folder = target / cat
            try:
                if folder.is_dir() and not any(folder.iterdir()):
                    folder.rmdir()
            except OSError:
                pass
        remaining = runs[:-1] + ([{"moves": list(reversed(left))}] if left else [])
        _write_manifest(manifest_path, target, remaining)
        return {"target": str(target), "restored": restored, "errors": errors}

    def delete_empty_dirs(self, target_dir: str | Path, dry_run: bool = True) -> dict[str, Any]:
        """Remove empty subdirectories, including ones that only contain empty
        subdirectories. Ignored folders (.git, venvs, node_modules...) are
        never entered or removed, and count as content for their parent.

        The dry run reports exactly what the real run removes.
        """
        target = _require_dir(target_dir)
        _assert_safe_target(target)

        order: list[Path] = []  # parents before children
        has_content: set[Path] = set()
        for root, dirs, files in os.walk(target):
            folder = Path(root)
            order.append(folder)
            # Skipped folders and links are content: we won't look inside them.
            kept = [d for d in dirs if d not in IGNORE_DIRS and not (folder / d).is_symlink()]
            if files or len(kept) < len(dirs):
                has_content.add(folder)
            dirs[:] = kept

        removed: list[str] = []
        for folder in reversed(order):  # deepest first, so emptiness propagates up
            if folder == target:
                continue
            if folder in has_content:
                has_content.add(folder.parent)
            else:
                removed.append(str(folder))

        if not dry_run:
            done = []
            for path in removed:
                try:
                    Path(path).rmdir()
                    done.append(path)
                except OSError:
                    continue  # something appeared since the scan; leave it
            removed = done

        return {"target": str(target), "dry_run": dry_run, "removed_count": len(removed), "removed": removed}
