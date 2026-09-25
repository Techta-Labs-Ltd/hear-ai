from __future__ import annotations

import hashlib
import shutil
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
    def _safe_component(value: str) -> str:
        raw = str(value)
        readable = "".join(
            character if character.isalnum() or character in "_.-" else "_"
            for character in raw
        )[:64]
        digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
        return f"{readable or 'item'}-{digest}"