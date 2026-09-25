from enum import StrEnum


class ExecutionErrorCode(StrEnum):
    CANCELLED = "cancelled"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    ENGINE_UNAVAILABLE = "engine_unavailable"
    INVALID_REQUEST = "invalid_request"
    INVALID_AUDIO = "invalid_audio"
    SOURCE_MISMATCH = "source_mismatch"
    STORAGE_FAILED = "storage_failed"
    PROCESS_FAILED = "process_failed"


class ExecutionError(RuntimeError):
    def __init__(
        self,
        code: ExecutionErrorCode,
        message: str,
        *,
        worker_restart_required: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.worker_restart_required = worker_restart_required
