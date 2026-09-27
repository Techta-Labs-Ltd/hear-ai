import hashlib
import importlib.metadata
import os
import stat
import threading
from collections.abc import Callable
from pathlib import Path

from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


class PinnedAssetProbe:
    CHUNK_BYTES = 1024 * 1024
    OPEN_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK

    @classmethod
    def read_regular(cls, path: Path, *, maximum_bytes: int) -> bytes:
        if not path.is_absolute() or maximum_bytes <= 0:
            raise ValueError("pinned cleaner asset path or limit is invalid")
        try:
            if path.resolve(strict=True) != path:
                raise OSError("asset traverses symlink")
            descriptor = os.open(path, cls.OPEN_FLAGS)
        except OSError:
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE,
                "pinned cleaner asset unavailable",
            ) from None
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum_bytes:
                raise CleanExecutionError(
                    ErrorCode.ENGINE_UNAVAILABLE,
                    "pinned cleaner asset invalid",
                )
            payload = bytearray()
            while len(payload) <= maximum_bytes:
                chunk = os.read(descriptor, min(cls.CHUNK_BYTES, maximum_bytes + 1 - len(payload)))
                if not chunk:
                    break
                payload.extend(chunk)
            after = os.fstat(descriptor)
            if len(payload) != before.st_size or (after.st_dev, after.st_ino, after.st_size) != (
                before.st_dev,
                before.st_ino,
                before.st_size,
            ):
                raise CleanExecutionError(
                    ErrorCode.ENGINE_UNAVAILABLE,
                    "pinned cleaner asset changed while reading",
                )
            return bytes(payload)
        except OSError:
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE,
                "pinned cleaner asset read failed",
            ) from None
        finally:
            os.close(descriptor)

    @staticmethod
    def directory(path: Path) -> os.stat_result:
        try:
            if path.resolve(strict=True) != path:
                raise OSError("asset directory traverses symlink")
            descriptor = os.open(
                path,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NONBLOCK,
            )
        except OSError:
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE,
                "pinned cleaner asset directory unavailable",
            ) from None
        try:
            info = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        if not stat.S_ISDIR(info.st_mode):
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE,
                "pinned cleaner asset directory invalid",
            )
        return info

    @classmethod
    def regular(cls, path: Path, *, maximum_bytes: int | None = None) -> os.stat_result:
        try:
            if path.resolve(strict=True) != path:
                raise OSError("asset traverses symlink")
            descriptor = os.open(path, cls.OPEN_FLAGS)
        except OSError:
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE,
                "pinned cleaner asset unavailable",
            ) from None
        try:
            info = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_size <= 0
            or (maximum_bytes is not None and info.st_size > maximum_bytes)
        ):
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE,
                "pinned cleaner asset invalid",
            )
        return info

    @classmethod
    def sha256(
        cls,
        path: Path,
        expected: str,
        *,
        maximum_bytes: int | None = None,
        check: Callable[[], None] | None = None,
    ) -> os.stat_result:
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(value not in "0123456789abcdef" for value in expected)
        ):
            raise ValueError("pinned cleaner asset requires SHA-256")
        info = cls.regular(path, maximum_bytes=maximum_bytes)
        digest = hashlib.sha256()
        try:
            descriptor = os.open(path, cls.OPEN_FLAGS)
        except OSError:
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE,
                "pinned cleaner asset unavailable",
            ) from None
        try:
            before = os.fstat(descriptor)
            while True:
                if check is not None:
                    check()
                chunk = os.read(descriptor, cls.CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
            after = os.fstat(descriptor)
        except OSError:
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE,
                "pinned cleaner asset verification failed",
            ) from None
        finally:
            os.close(descriptor)
        identity_before = before.st_dev, before.st_ino, before.st_size
        identity_after = after.st_dev, after.st_ino, after.st_size
        identity_initial = info.st_dev, info.st_ino, info.st_size
        if (
            identity_before != identity_initial
            or identity_after != identity_before
            or digest.hexdigest() != expected
        ):
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE,
                "pinned cleaner asset mismatch",
            )
        return after

    @staticmethod
    def packages(versions: tuple[tuple[str, str], ...] | dict[str, str]) -> None:
        entries = versions.items() if isinstance(versions, dict) else versions
        for package, expected in entries:
            try:
                actual = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                raise CleanExecutionError(
                    ErrorCode.ENGINE_UNAVAILABLE,
                    "cleaner runtime dependency missing",
                ) from None
            if actual != expected:
                raise CleanExecutionError(
                    ErrorCode.ENGINE_UNAVAILABLE,
                    "cleaner runtime dependency mismatch",
                )


class PinnedAssetSet:
    def __init__(self, assets: tuple[tuple[Path, str, int | None], ...]):
        if not assets or len({path for path, _, _ in assets}) != len(assets):
            raise ValueError("pinned cleaner assets must be non-empty and unique")
        self._assets = assets
        self._verified: dict[Path, tuple[int, int, int, int, int]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )

    def verify(self) -> None:
        with self._lock:
            for path, expected, maximum_bytes in self._assets:
                current = PinnedAssetProbe.regular(path, maximum_bytes=maximum_bytes)
                identity = self._identity(current)
                if self._verified.get(path) == identity:
                    continue
                verified = PinnedAssetProbe.sha256(
                    path,
                    expected,
                    maximum_bytes=maximum_bytes,
                )
                self._verified[path] = self._identity(verified)
