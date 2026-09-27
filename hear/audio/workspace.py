from __future__ import annotations

import hashlib
import os
import shutil
import time
from pathlib import Path


class AudioWorkspace:
    def __init__(
        self,
        root: Path,
        job_id: str,
        attempt_id: str,
    ) -> None:
        self._root = root
        self._path = (
            root
            / "jobs"
            / self._safe_component(job_id)
            / self._safe_component(attempt_id)
        )

    @property
    def path(self) -> Path:
        self._path.mkdir(parents=True, exist_ok=True)
        return self._path

    def file(self, name: str) -> Path:
        if not name or "/" in name or "\\" in name or name in {".", ".."}:
            raise ValueError("invalid workspace filename")
        return self.path / name

    def cleanup(self) -> None:
        shutil.rmtree(self._path, ignore_errors=True)
        parent = self._path.parent
        try:
            parent.rmdir()
        except OSError:
            pass

    @staticmethod
    def sweep(root: Path, max_age_seconds: float) -> dict[str, int]:
        if max_age_seconds <= 0:
            raise ValueError("workspace retention must be positive")
        cutoff = time.time() - max_age_seconds
        removed = 0
        bytes_freed = 0
        root.mkdir(parents=True, exist_ok=True)
        jobs = root / "jobs"
        if jobs.is_dir() and not jobs.is_symlink():
            for job in jobs.iterdir():
                if not job.is_dir() or job.is_symlink():
                    continue
                for attempt in job.iterdir():
                    try:
                        if attempt.is_symlink() or not attempt.is_dir():
                            continue
                        if attempt.stat(follow_symlinks=False).st_mtime >= cutoff:
                            continue
                        bytes_freed += AudioWorkspace._path_size(attempt)
                        shutil.rmtree(attempt)
                        removed += 1
                    except OSError:
                        continue
                try:
                    job.rmdir()
                except OSError:
                    pass
        for entry in root.iterdir():
            try:
                if entry == jobs or entry.is_dir() or entry.is_symlink():
                    continue
                info = entry.stat(follow_symlinks=False)
                if info.st_mtime >= cutoff:
                    continue
                bytes_freed += info.st_size
                entry.unlink()
                removed += 1
            except OSError:
                continue
        return {"removed": removed, "bytes_freed": bytes_freed}

    @staticmethod
    def purge(root: Path) -> dict[str, int]:
        root.mkdir(parents=True, exist_ok=True)
        removed = 0
        bytes_freed = 0
        for entry in root.iterdir():
            try:
                bytes_freed += AudioWorkspace._path_size(entry)
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
                removed += 1
            except OSError:
                continue
        return {"removed": removed, "bytes_freed": bytes_freed}

    @staticmethod
    def _path_size(path: Path) -> int:
        try:
            if path.is_symlink():
                return 0
            if path.is_file() and not path.is_symlink():
                return path.stat(follow_symlinks=False).st_size
            total = 0
            for directory, _, files in os.walk(path, followlinks=False):
                for filename in files:
                    entry = Path(directory) / filename
                    try:
                        if not entry.is_symlink():
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
            return total
        except OSError:
            return 0

    @staticmethod
    def _safe_component(value: str) -> str:
        raw = str(value)
        readable = "".join(
            character if character.isalnum() or character in "_.-" else "_"
            for character in raw
        )[:64]
        digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
        return f"{readable or 'item'}-{digest}"
