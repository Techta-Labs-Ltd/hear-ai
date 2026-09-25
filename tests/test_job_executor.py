from collections.abc import AsyncIterator

import pytest

from hear.contracts.events import ExecutionEvent, ExecutionEventType
from hear.contracts.jobs import AttemptEnvelope, JobType, WorkerRole
from hear.execution.context import ExecutionContext
from hear.execution.executor import JobExecutor
from hear.runtime.roles import RoleRegistry


class FakeWorkflow:
    async def execute(self, context: ExecutionContext) -> AsyncIterator[ExecutionEvent]:
        yield context.event(
            ExecutionEventType.STAGE,
            stage="transcribing",
            progress_pct=50.0,
        )
        yield context.event(
            ExecutionEventType.OUTCOME,
            stage="completed",
            progress_pct=100.0,
        )


def envelope(job_type: JobType) -> AttemptEnvelope:
    return AttemptEnvelope.model_validate(
        {
            "job_id": "job-1",
            "run_id": "run-1",
            "attempt_id": "attempt-1",
            "job_type": job_type,
            "track_id": "track-1",
            "user_id": "user-1",
            "source": {"url": "https://example.invalid/audio.mp3", "revision": 1},
            "storage": {
                "reference": "storage-1",
                "token": "secret",
                "expires_at": "2026-09-26T00:00:00Z",
            },
            "reporting": {
                "backend_base_url": "https://api.example.invalid",
                "token": "secret",
            },
        }
    )


@pytest.mark.asyncio
async def test_executor_streams_monotonic_events():
    executor = JobExecutor(
        RoleRegistry().get(WorkerRole.PIPELINE),
        {JobType.PIPELINE: FakeWorkflow()},
    )
    events = [event async for event in executor.stream(envelope(JobType.PIPELINE))]
    assert [event.sequence for event in events] == [1, 2, 3]
    assert [event.event for event in events] == [
        ExecutionEventType.STARTED,
        ExecutionEventType.STAGE,
        ExecutionEventType.OUTCOME,
    ]


@pytest.mark.asyncio
async def test_executor_rejects_wrong_role():
    executor = JobExecutor(
        RoleRegistry().get(WorkerRole.TRANSCRIPTION),
        {JobType.TRANSCRIPTION: FakeWorkflow()},
    )
    with pytest.raises(RuntimeError, match="worker_capability_mismatch"):
        _ = [event async for event in executor.stream(envelope(JobType.PIPELINE))]
