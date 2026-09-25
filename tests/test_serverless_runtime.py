from collections.abc import AsyncIterator

import pytest

from hear.contracts.events import ExecutionEvent, ExecutionEventType
from hear.contracts.jobs import JobType, WorkerRole
from hear.execution.context import ExecutionContext
from hear.execution.executor import JobExecutor
from hear.runtime.roles import RoleRegistry
from hear.runtime.serverless import ServerlessRuntime


class FakeWorkflow:
    async def execute(self, context: ExecutionContext) -> AsyncIterator[ExecutionEvent]:
        yield context.event(ExecutionEventType.PROGRESS, stage="work", progress_pct=75.0)


@pytest.mark.asyncio
async def test_serverless_runtime_uses_shared_executor():
    runtime = ServerlessRuntime(
        JobExecutor(
            RoleRegistry().get(WorkerRole.TRANSCRIPTION),
            {JobType.TRANSCRIPTION: FakeWorkflow()},
        )
    )
    payload = {
        "job_id": "job-1",
        "run_id": "run-1",
        "attempt_id": "attempt-1",
        "job_type": "transcription",
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
    events = [event async for event in runtime.stream(payload)]
    assert [event["event"] for event in events] == ["started", "progress"]
    assert [event["sequence"] for event in events] == [1, 2]
