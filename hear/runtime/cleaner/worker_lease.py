"""Linux host-local worker ownership, held for the entire model-cache lifetime.

All cleaner containers on a device must mount the same local lock directory.
This is not a VRAM partition or admission for unrelated transcription/TTS actors.
Never unlink a lock on release: replacement inodes would permit split ownership.
"""
import fcntl
import os
import stat
import threading
from pathlib import Path
from typing import Literal
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode

class WorkerLease:
    def __init__(self, directory: Path, lane: Literal["gpu", "cpu"]):
        if lane not in ("gpu", "cpu"):
            raise ValueError("invalid cleaner lane")
        if not directory.is_absolute() or directory.resolve() != directory:
            raise ValueError("worker lock directory must be absolute without symlinks")
        self.path = directory / f"cleaner-{lane}.lock"
        self.lane = lane
        self.pid = os.getpid()
        self.fd = None
        self._attempt_lock = threading.Lock()
        self._unhealthy = False
        descriptor = None
        try:
            descriptor = os.open(
                self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600
            )
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise OSError("invalid lock inode")
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.fd = descriptor
            self.assert_owned(lane)
        except OSError as exc:
            if descriptor is not None:
                os.close(descriptor)
            self.fd = None
            raise CleanExecutionError(
                ErrorCode.RESOURCE_EXHAUSTED, "cleaner worker lane unavailable"
            ) from exc
        except BaseException:
            if descriptor is not None:
                os.close(descriptor)
            self.fd = None
            raise

    def assert_owned(self, lane: str) -> None:
        if self._unhealthy:
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE,
                "cleaner worker requires process restart",
                worker_restart_required=True,
            )
        if self.fd is None or self.pid != os.getpid() or self.lane != lane:
            raise CleanExecutionError(ErrorCode.RESOURCE_EXHAUSTED, "worker lane is not owned")
        try:
            current = self.path.lstat()
            held = os.fstat(self.fd)
            if (current.st_dev, current.st_ino) != (held.st_dev, held.st_ino):
                raise OSError("worker lock inode changed")
        except OSError as exc:
            raise CleanExecutionError(
                ErrorCode.RESOURCE_EXHAUSTED, "worker ownership changed"
            ) from exc

    def mark_unhealthy(self) -> None:
        """Retain ownership but stop admission; only process replacement recovers."""
        self._unhealthy = True

    def close(self) -> None:
        """Call only after worker/model teardown; never after each job."""
        if not self._attempt_lock.acquire(blocking=False):
            raise CleanExecutionError(ErrorCode.RESOURCE_EXHAUSTED, "worker still has active work")
        try:
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None
        finally:
            self._attempt_lock.release()

    def attempt(self, lane: str):
        return WorkerAttempt(self, lane)

    def __enter__(self):
        self.assert_owned(self.lane)
        return self

    def __exit__(self, *args):
        self.close()


class WorkerAttempt:
    def __init__(self, worker: WorkerLease, lane: str):
        self.worker = worker
        self.lane = lane
        self.active = False

    def __enter__(self):
        if not self.worker._attempt_lock.acquire(blocking=False):
            raise CleanExecutionError(ErrorCode.RESOURCE_EXHAUSTED, "worker attempt lane occupied")
        try:
            self.worker.assert_owned(self.lane)
            self.active = True
            return self
        except BaseException:
            self.worker._attempt_lock.release()
            raise

    def __exit__(self, *args):
        if self.active:
            self.active = False
            self.worker._attempt_lock.release()
