import stat
from dataclasses import dataclass
from pathlib import Path

from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


@dataclass(frozen=True)
class OwnedFileIdentity:
    device: int
    inode: int


class AttemptPathCleanup:
    POLICY = "attempt-owned-path-cleanup-v1"

    @staticmethod
    def identity(path: Path) -> OwnedFileIdentity:
        info = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(info.st_mode):
            raise OSError("attempt output is not a regular file")
        return OwnedFileIdentity(info.st_dev, info.st_ino)

    @staticmethod
    def _code(primary: BaseException | None, fallback: ErrorCode) -> ErrorCode:
        if isinstance(primary, CleanExecutionError):
            return primary.code
        if isinstance(primary, MemoryError):
            return ErrorCode.RESOURCE_EXHAUSTED
        return fallback

    @classmethod
    def _failure(
        cls,
        primary: BaseException | None,
        fallback: ErrorCode,
        message: str,
    ) -> CleanExecutionError:
        return CleanExecutionError(
            cls._code(primary, fallback), message, worker_restart_required=True
        )

    @classmethod
    def remove_private(
        cls,
        path: Path,
        primary: BaseException | None,
        *,
        fallback: ErrorCode = ErrorCode.PROCESS_FAILED,
        message: str = "attempt file cleanup failed; worker requires process restart",
    ) -> None:
        try:
            info = path.lstat()
        except FileNotFoundError:
            return
        except OSError:
            raise cls._failure(primary, fallback, message) from primary
        if path.is_symlink() or not stat.S_ISREG(info.st_mode):
            raise cls._failure(primary, fallback, message) from primary
        try:
            path.unlink()
        except OSError:
            raise cls._failure(primary, fallback, message) from primary

    @classmethod
    def remove_if_owned(
        cls,
        path: Path,
        identity: OwnedFileIdentity,
        primary: BaseException | None,
        *,
        fallback: ErrorCode = ErrorCode.PROCESS_FAILED,
        message: str = "attempt output cleanup failed; worker requires process restart",
    ) -> None:
        try:
            info = path.lstat()
        except FileNotFoundError:
            return
        except OSError:
            raise cls._failure(primary, fallback, message) from primary
        if (info.st_dev, info.st_ino) != (identity.device, identity.inode):
            return
        if path.is_symlink() or not stat.S_ISREG(info.st_mode):
            raise cls._failure(primary, fallback, message) from primary
        try:
            path.unlink()
        except OSError:
            raise cls._failure(primary, fallback, message) from primary
