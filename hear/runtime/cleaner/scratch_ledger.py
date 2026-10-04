"""Host-wide scratch disk admission shared by every worker process on one machine.

Each job's ResourceGuard already caps its own workspace. That alone lets several
long jobs together promise more disk than exists, and a full disk fails jobs
mid-write and blocks RabbitMQ. The ledger records each live workspace's total
reservation in a file under the scratch root, and admits a reservation only if
free space still covers every other job's unwritten remainder plus a floor.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path

from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


class HostScratchLedger:
    def __init__(self, root: Path, *, min_free_bytes: int) -> None:
        if min_free_bytes < 0:
            raise ValueError("min_free_bytes must be non-negative")
        self._root = root
        self._min_free_bytes = min_free_bytes
        self._state = root / ".scratch-ledger.json"
        self._lock = root / ".scratch-ledger.lock"

    @staticmethod
    def _written(workspace: Path) -> int:
        if not workspace.is_dir():
            return 0
        return sum(p.stat().st_size for p in workspace.rglob("*") if p.is_file())

    @staticmethod
    def _alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _live_entries(self) -> dict[str, dict]:
        try:
            entries = json.loads(self._state.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        return {
            path: entry
            for path, entry in entries.items()
            if Path(path).is_dir() and self._alive(int(entry["pid"]))
        }

    def reserve(
        self,
        workspace: Path,
        total_bytes: int,
        *,
        wait: Callable[[], None] | None = None,
        poll_seconds: float = 5.0,
    ) -> None:
        """Hold `total_bytes` of disk for `workspace` (its whole footprint, not a delta).

        With `wait`, a job blocked only by other jobs' reservations queues here instead
        of failing; `wait` must raise when the attempt is cancelled or out of time.
        """
        while True:
            try:
                return self._admit(workspace, total_bytes)
            except CleanExecutionError as busy:
                if wait is None or not busy.retryable:
                    raise
            wait()
            time.sleep(poll_seconds)

    def _admit(self, workspace: Path, total_bytes: int) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        key = str(workspace.resolve())
        with self._lock.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            entries = self._live_entries()
            others_written = 0
            others_outstanding = 0
            for path, entry in entries.items():
                if path == key:
                    continue
                written = self._written(Path(path))
                others_written += written
                others_outstanding += max(0, int(entry["reserved"]) - written)
            need = max(0, total_bytes - self._written(workspace))
            free = shutil.disk_usage(self._root).free
            # Other jobs' files are freed when they finish, so judge "too big" on the idle host.
            if need > free + others_written - self._min_free_bytes:
                # Too large for this host even when idle: retrying cannot help.
                raise CleanExecutionError(
                    ErrorCode.RESOURCE_EXHAUSTED, "input needs more scratch disk than this host has"
                )
            if need > free - others_outstanding - self._min_free_bytes:
                raise CleanExecutionError(
                    ErrorCode.RESOURCE_EXHAUSTED,
                    "scratch disk is held by other jobs",
                    retryable=True,
                )
            previous = int(entries.get(key, {}).get("reserved", 0))
            entries[key] = {"pid": os.getpid(), "reserved": max(previous, total_bytes)}
            temporary = self._state.with_suffix(".tmp")
            temporary.write_text(json.dumps(entries))
            os.replace(temporary, self._state)
