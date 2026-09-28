from __future__ import annotations

import fcntl
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class HostJobPermit:
    descriptor: int
    closed: bool = False
    extra_descriptors: tuple[int, ...] = ()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
        finally:
            os.close(self.descriptor)
            for fd in self.extra_descriptors:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)


class HostJobAdmission:
    def __init__(
        self, lock_path: Path, max_jobs: int = 1, role: str | None = None, role_limit: int = 1
    ) -> None:
        if not 1 <= max_jobs <= 16:
            raise ValueError("invalid_host_job_limit")
        self._lock_path = lock_path
        self._max_jobs = max_jobs
        if role is not None and (
            not role or any(c not in "abcdefghijklmnopqrstuvwxyz_" for c in role)
        ):
            raise ValueError("invalid_admission_role")
        if not 1 <= role_limit <= max_jobs:
            raise ValueError("invalid_role_job_limit")
        self._role = role
        self._role_limit = role_limit

    def try_acquire(self) -> HostJobPermit | None:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        for slot in range(self._max_jobs):
            path = (
                self._lock_path
                if slot == 0
                else self._lock_path.with_name(self._lock_path.name + f".{slot}")
            )
            descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(descriptor)
                continue
            if self._role is None:
                return HostJobPermit(descriptor)
            for index in range(self._role_limit):
                role_path = self._lock_path.with_name(
                    self._lock_path.name + f".{self._role}.{index}"
                )
                role_fd = os.open(role_path, os.O_CREAT | os.O_RDWR, 0o600)
                try:
                    fcntl.flock(role_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(role_fd)
                    continue
                return HostJobPermit(descriptor, extra_descriptors=(role_fd,))
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
            return None
        return None
