from __future__ import annotations

import fcntl
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class HostJobPermit:
    descriptor: int
    closed: bool = False

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
        finally:
            os.close(self.descriptor)


class HostJobAdmission:
    def __init__(self, lock_path: Path) -> None:
        self._lock_path = lock_path

    def try_acquire(self) -> HostJobPermit | None:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(descriptor)
            return None
        return HostJobPermit(descriptor)
