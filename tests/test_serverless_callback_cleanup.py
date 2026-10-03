from types import SimpleNamespace

import httpx
import pytest

from hear.contracts.events import ExecutionEvent, ExecutionEventType
from hear.contracts.jobs import AttemptClaim, ClaimDecision, JobType
from hear.execution.executor import JobExecutor
from hear.runtime.roles import WorkerRole
from hear.runtime.serverless import ServerlessRuntime
from tests.test_job_executor_v2 import TestJobExecutor


@pytest.mark.anyio
async def test_backend_callback_failure_closes_workflow_before_releasing_admission():
    cleaned = []
    envelope = TestJobExecutor.envelope()

    class Workflow:
        async def stream(self, envelope):
            try:
                yield ExecutionEvent(
                    event_id="event",
                    job_id=envelope.job_id,
                    attempt_id=envelope.attempt_id,
                    track_id=envelope.track_id,
                    job_type=envelope.job_type,
                    source_revision=1,
                    sequence=1,
                    event=ExecutionEventType.PROGRESS,
                )
            finally:
                cleaned.append(True)

    class Backend:
        async def claim(self, envelope):
            return AttemptClaim(decision=ClaimDecision.EXECUTE)

        async def heartbeat(self, *args):
            pass

        async def event(self, *args):
            response = httpx.Response(
                403, request=httpx.Request("POST", "https://backend.test/events")
            )
            response.raise_for_status()

    runtime = ServerlessRuntime(
        WorkerRole.TRANSCRIPTION,
        JobExecutor({JobType.TRANSCRIPTION: Workflow()}),
        Backend(),
        SimpleNamespace(serverless=SimpleNamespace(progress_update=lambda *args: None)),
    )
    with pytest.raises(httpx.HTTPStatusError):
        _ = [
            event
            async for event in runtime.handler(
                {"id": "job", "input": envelope.model_dump(mode="json")}
            )
        ]
    assert cleaned == [True]
    assert runtime._admission._value == 1


@pytest.mark.parametrize("status,rejected", [(422, True), (403, True), (503, False)])
@pytest.mark.anyio
async def test_backend_claim_verdicts_become_rejection_events_like_the_pod(status, rejected):
    envelope = TestJobExecutor.envelope()

    class Backend:
        async def claim(self, envelope):
            response = httpx.Response(
                status, request=httpx.Request("POST", "https://backend.test/claim")
            )
            response.raise_for_status()

    runtime = ServerlessRuntime(
        WorkerRole.TRANSCRIPTION,
        JobExecutor({JobType.TRANSCRIPTION: object()}),
        Backend(),
        SimpleNamespace(serverless=SimpleNamespace(progress_update=lambda *args: None)),
    )
    job = {"id": "job", "input": envelope.model_dump(mode="json")}
    if rejected:
        events = [event async for event in runtime.handler(job)]
        assert events == [
            {
                "event": "backend_attempt_rejected",
                "job_id": envelope.job_id,
                "attempt_id": envelope.attempt_id,
                "http_status": status,
            }
        ]
    else:
        with pytest.raises(httpx.HTTPStatusError):
            _ = [event async for event in runtime.handler(job)]
    assert runtime._admission._value == 1
