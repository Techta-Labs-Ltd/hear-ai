from __future__ import annotations

import logging
import re
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol

import httpx

from hear.contracts.events import ExecutionEvent, ExecutionEventType
from hear.contracts.jobs import AttemptEnvelope, JobType
from hear.contracts.outcomes import ExecutionOutcome
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode

logger = logging.getLogger(__name__)


class ExecutionWorkflow(Protocol):
    def stream(self, envelope: AttemptEnvelope) -> AsyncIterator[ExecutionEvent]: ...


class FailurePolicy:
    """Decide whether a workflow error is final for this input or worth another attempt.

    Final errors become a `failed` outcome the backend records at once; anything that
    may succeed on a fresh attempt (engine/storage hiccups, disk held by other jobs, a
    faulted GPU worker, unknown errors) is re-raised so the transport retries it.
    """

    TRANSIENT_CODES = frozenset({ErrorCode.ENGINE_UNAVAILABLE, ErrorCode.STORAGE_FAILED})

    @classmethod
    def final_error_code(cls, error: Exception) -> str | None:
        if isinstance(error, CleanExecutionError):
            if error.retryable or error.worker_restart_required or error.code in cls.TRANSIENT_CODES:
                return None
            return error.code.value
        if isinstance(error, httpx.HTTPStatusError):
            status = error.response.status_code
            return "source_unavailable" if 400 <= status < 500 and status not in (408, 429) else None
        if isinstance(error, ValueError):
            return "invalid_request"
        return None


class WorkerFailure(RuntimeError):
    """A failure the transport reports; its message is a FailureSummary, not a traceback."""


class FailureSummary:
    """The exception type and message first, then where it was raised.

    Transports and callbacks keep only the first few hundred characters. A chained
    traceback leads with the worker process's frames and loses the actual error,
    so the summary puts the error first and keeps only the innermost frame.
    """

    LIMIT = 500
    _REMOTE_FRAME = re.compile(r'File "([^"]+)", line (\d+), in (\S+)')

    @classmethod
    def describe(cls, error: BaseException) -> str:
        message = f"{type(error).__name__}: {error}".strip().rstrip(":")
        origin = cls._origin(error)
        suffix = f" (at {origin})" if origin else ""
        return message[: cls.LIMIT - len(suffix)] + suffix

    @classmethod
    def _origin(cls, error: BaseException) -> str:
        # A process-pool error carries the worker's traceback as text on its cause.
        remote = getattr(error.__cause__, "tb", None)
        if isinstance(remote, str):
            frames = cls._REMOTE_FRAME.findall(remote)
            if frames:
                path, line, function = frames[-1]
                return f"{Path(path).name}:{line} in {function}"
        frame = error.__traceback__
        if frame is None:
            return ""
        while frame.tb_next is not None:
            frame = frame.tb_next
        code = frame.tb_frame.f_code
        return f"{Path(code.co_filename).name}:{frame.tb_lineno} in {code.co_name}"


class JobExecutor:
    def __init__(self, workflows: dict[JobType, ExecutionWorkflow]) -> None:
        self._workflows = dict(workflows)

    async def stream(self, envelope: AttemptEnvelope) -> AsyncIterator[ExecutionEvent]:
        workflow = self._workflows.get(envelope.job_type)
        if workflow is None:
            raise RuntimeError(f"workflow_unavailable:{envelope.job_type.value}")
        iterator = workflow.stream(envelope)
        sequence = 0
        reported = False
        try:
            async for event in iterator:
                sequence = max(sequence, event.sequence)
                reported = reported or event.event == ExecutionEventType.OUTCOME
                yield event
        except Exception as error:
            code = None if reported else FailurePolicy.final_error_code(error)
            logger.exception(
                "attempt_failed job_id=%s attempt_id=%s job_type=%s final=%s",
                envelope.job_id,
                envelope.attempt_id,
                envelope.job_type.value,
                code or ("already_reported" if reported else "retry"),
            )
            if reported:
                return
            if code is None:
                raise
            yield self._failed(envelope, sequence + 1, code, FailureSummary.describe(error))
        finally:
            close = getattr(iterator, "aclose", None)
            if callable(close):
                await close()

    @staticmethod
    def _failed(envelope: AttemptEnvelope, sequence: int, code: str, message: str) -> ExecutionEvent:
        outcome = ExecutionOutcome(
            job_id=envelope.job_id,
            attempt_id=envelope.attempt_id,
            track_id=envelope.track_id,
            job_type=envelope.job_type,
            backend_id=envelope.backend_id,
            source_revision=envelope.source.revision,
            status="failed",
            error_code=code,
            result={"message": message[:500]},
        )
        return ExecutionEvent(
            event_id=str(uuid.uuid4()),
            job_id=envelope.job_id,
            attempt_id=envelope.attempt_id,
            track_id=envelope.track_id,
            job_type=envelope.job_type,
            backend_id=envelope.backend_id,
            source_revision=envelope.source.revision,
            sequence=sequence,
            event=ExecutionEventType.OUTCOME,
            stage="failed",
            progress_pct=100,
            data={"outcome": outcome.model_dump(mode="json")},
        )
