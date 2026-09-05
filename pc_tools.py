"""
pc_tools.py
Local PC Optimization and Automation Engine for NEXUS AI.
Provides safe operations for directory analysis, folder cleanup,
duplicate file detection, and large file scanning.
"""

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

# Directories that are noise for a duplicate / large-file / usage scan:
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


def _walk(target: Path):
    """os.walk that prunes IGNORE_DIRS in place."""
    for root, dirs, files in os.walk(target):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        yield root, dirs, files


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

    @staticmethod
    def file_hash(filepath: str | Path, block_size: int = 65536) -> str:
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            for block in iter(lambda: f.read(block_size), b""):
                hasher.update(block)
        return hasher.hexdigest()

    def analyze_directory(self, target_dir: str | Path) -> dict[str, Any]:
        target = Path(target_dir).resolve()
        if not target.exists() or not target.is_dir():
            raise ValueError(f"Directory non-existent or invalid: {target}")

        total_files = 0
        total_size = 0
        categories: dict[str, int] = {cat: 0 for cat in self.CATEGORY_MAP}
        categories["Others"] = 0

        for root, _, files in _walk(target):
            for fname in files:
                total_files += 1
                fpath = Path(root) / fname
                try:
                    fsize = fpath.stat().st_size
                    total_size += fsize
                except Exception:
                    continue

                ext = fpath.suffix.lower()
                matched = False
                for cat, exts in self.CATEGORY_MAP.items():
                    if ext in exts:
                        categories[cat] += 1
                        matched = True
                        break
                if not matched:
                    categories["Others"] += 1

        return {
            "path": str(target),
            "total_files": total_files,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "category_counts": categories,
        }

    def find_duplicates(self, target_dir: str | Path) -> list[dict[str, Any]]:
        target = Path(target_dir).resolve()
        if not target.exists() or not target.is_dir():
            raise ValueError(f"Directory non-existent or invalid: {target}")

        size_map: dict[int, list[Path]] = {}
        for root, _, files in _walk(target):
            for fname in files:
                fpath = Path(root) / fname
                try:
                    size = fpath.stat().st_size
                    if size > 0:
                        size_map.setdefault(size, []).append(fpath)
                except Exception:
                    continue

        duplicates = []
        for size, paths in size_map.items():
            if len(paths) < 2:
                continue
            hash_map: dict[str, list[str]] = {}
            for path in paths:
                try:
                    h = self.file_hash(path)
                    hash_map.setdefault(h, []).append(str(path))
                except Exception:
                    continue
            for h, dup_paths in hash_map.items():
                if len(dup_paths) > 1:
                    duplicates.append({
                        "hash": h[:12],
                        "size_mb": round(size / (1024 * 1024), 2),
                        "paths": dup_paths,
                    })

        return duplicates

    def find_large_files(self, target_dir: str | Path, min_size_mb: float = 50.0) -> list[dict[str, Any]]:
        target = Path(target_dir).resolve()
        if not target.exists() or not target.is_dir():
            raise ValueError(f"Directory non-existent or invalid: {target}")

        threshold_bytes = min_size_mb * 1024 * 1024
        large_files = []

        for root, _, files in _walk(target):
            for fname in files:
                fpath = Path(root) / fname
                try:
                    size = fpath.stat().st_size
                    if size >= threshold_bytes:
                        large_files.append({
                            "path": str(fpath),
                            "size_mb": round(size / (1024 * 1024), 2),
                        })
                except Exception:
                    continue

        large_files.sort(key=lambda x: x["size_mb"], reverse=True)
        return large_files

    def _category_for(self, ext: str) -> str:
        for cat, exts in self.CATEGORY_MAP.items():
            if ext in exts:
                return cat
        return "Others"

    def organize_folder(self, target_dir: str | Path, dry_run: bool = True) -> dict[str, Any]:
        """Move loose files in `target_dir` into category subfolders.

        Only the top level is touched. Hidden/dotfiles are left alone.
        When `dry_run` is False a manifest is written so the move can be
        undone with `undo_last_organize`.
        """
        target = Path(target_dir).resolve()
        if not target.exists() or not target.is_dir():
            raise ValueError(f"Directory non-existent or invalid: {target}")
        _assert_safe_target(target)

        category_names = set(self.CATEGORY_MAP) | {"Others"}
        moves: list[dict[str, str]] = []

        for fname in sorted(os.listdir(target)):
            fpath = target / fname
            if fpath.is_dir():
                continue
            if fname.startswith(".") or fname == MANIFEST_NAME:
                continue  # leave hidden / config files where they are

            category = self._category_for(fpath.suffix.lower())
            dest_folder = target / category
            dest_file = dest_folder / fname

            # Collision-safe: never overwrite an existing file.
            if dest_file.exists():
                n = 1
                while (dest_folder / f"{fpath.stem} ({n}){fpath.suffix}").exists():
                    n += 1
                dest_file = dest_folder / f"{fpath.stem} ({n}){fpath.suffix}"

            moves.append({
                "source": str(fpath),
                "destination": str(dest_file),
                "category": category,
            })

            if not dry_run:
                dest_folder.mkdir(exist_ok=True)
                shutil.move(str(fpath), str(dest_file))

        manifest_path = target / MANIFEST_NAME
        if not dry_run and moves:
            manifest_path.write_text(
                json.dumps({"target": str(target), "moves": moves}, indent=2),
                encoding="utf-8",
            )

        return {
            "target": str(target),
            "dry_run": dry_run,
            "moved_count": len(moves),
            "planned_moves": moves,
            "manifest": str(manifest_path) if (not dry_run and moves) else None,
        }

    def undo_last_organize(self, target_dir: str | Path) -> dict[str, Any]:
        """Reverse the most recent `organize_folder` using its manifest."""
        target = Path(target_dir).resolve()
        manifest_path = target / MANIFEST_NAME
        if not manifest_path.exists():
            raise ValueError(f"No organize manifest in {target} — nothing to undo.")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        restored, errors = 0, []
        for mv in reversed(manifest.get("moves", [])):
            now_at = Path(mv["destination"])
            back_to = Path(mv["source"])
            try:
                if now_at.exists():
                    back_to.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(now_at), str(back_to))
                    restored += 1
            except Exception as exc:  # noqa: BLE001 - report, keep going
                errors.append(f"{now_at.name}: {exc}")

        # Drop category folders that are now empty, then the manifest.
        for cat in set(self.CATEGORY_MAP) | {"Others"}:
            folder = target / cat
            try:
                if folder.is_dir() and not any(folder.iterdir()):
                    folder.rmdir()
            except OSError:
                pass
        manifest_path.unlink(missing_ok=True)

        return {"target": str(target), "restored": restored, "errors": errors}

    def delete_empty_dirs(self, target_dir: str | Path, dry_run: bool = True) -> dict[str, Any]:
        """Remove empty subdirectories (bottom-up, so nested empties collapse)."""
        target = Path(target_dir).resolve()
        if not target.exists() or not target.is_dir():
            raise ValueError(f"Directory non-existent or invalid: {target}")
        _assert_safe_target(target)

        removed: list[str] = []
        for root, dirs, _ in os.walk(target, topdown=False):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            folder = Path(root)
            if folder == target:
                continue
            try:
                if not any(folder.iterdir()):
                    removed.append(str(folder))
                    if not dry_run:
                        folder.rmdir()
            except OSError:
                continue

        return {
            "target": str(target),
            "dry_run": dry_run,
            "removed_count": len(removed),
            "removed": removed,
        }
