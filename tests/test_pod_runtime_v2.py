import asyncio

import pytest

from hear.contracts.events import ExecutionEvent, ExecutionEventType
from hear.contracts.jobs import AttemptClaim, AttemptEnvelope, ClaimDecision, JobType
from hear.execution.executor import JobExecutor
from hear.runtime.attempt_stream import AttemptRejection
from hear.runtime.pod import PodRuntime, PodRuntimeBusy
from hear.runtime.roles import WorkerRole


class FakeReadiness:
    def __init__(self):
        self.draining = False
        self.checks = {}

    def add_check(self, name, check):
        self.checks[name] = check

    def set_draining(self, draining):
        self.draining = draining

    def is_ready(self):
        return not self.draining and all(check() for check in self.checks.values())

    def snapshot(self):
        return {"status": "ready" if self.is_ready() else "draining"}


class FakeBackend:
    async def claim(self, _envelope):
        return AttemptClaim(decision=ClaimDecision.EXECUTE)

    async def heartbeat(self, _envelope, _sequence):
        return None


class FakeWorkflow:
    async def stream(self, envelope):
        yield ExecutionEvent(
            event_id="event-1",
            job_id=envelope.job_id,
            attempt_id=envelope.attempt_id,
            track_id=envelope.track_id,
            job_type=envelope.job_type,
            source_revision=envelope.source.revision,
            sequence=1,
            event=ExecutionEventType.PROGRESS,
            stage="transcribing",
            progress_pct=20,
        )


def envelope(job_type=JobType.TRANSCRIPTION):
    return AttemptEnvelope.model_validate(
        {
            "job_id": "job-1",
            "run_id": "run-1",
            "attempt_id": "attempt-1",
            "job_type": job_type.value,
            "track_id": "track-1",
            "user_id": "user-1",
            "source": {"url": "https://example.com/audio.wav", "revision": 1},
            "storage": {
                "endpoint_url": "https://s3.example.com",
                "bucket_name": "bucket",
                "key_id": "key",
                "application_key": "secret",
                "folder_prefix": "users/user-1/jobs/",
                "public_base_url": "https://cdn.example.com/media",
                "expires_at": "2030-01-01T00:00:00Z",
            },
            "artifact_prefix": "jobs/job-1/attempt-1",
            "deadline": "2030-01-01T00:00:00Z",
            "reporting_grant": "grant",
            "backend_base_url": "https://api.example.com",
        }
    )


def create_runtime(*, api_key="pod-key", max_jobs=1):
    readiness = FakeReadiness()
    runtime = PodRuntime(
        readiness,
        WorkerRole.TRANSCRIPTION,
        JobExecutor({JobType.TRANSCRIPTION: FakeWorkflow()}),
        FakeBackend(),
        api_key=api_key,
        max_concurrent_jobs=max_jobs,
    )
    return runtime, readiness


@pytest.mark.anyio
async def test_start_requires_api_key_and_marks_http_admission_ready():
    runtime, readiness = create_runtime(api_key="")

    with pytest.raises(RuntimeError, match="runtime_not_ready"):
        await runtime.start()

    assert not readiness.is_ready()


@pytest.mark.anyio
async def test_drain_marks_runtime_unready_and_waits_for_active_attempts():
    runtime, readiness = create_runtime()
    await runtime.start()
    attempt = await runtime.prepare_attempt(envelope())

    draining = asyncio.create_task(runtime.drain())
    await asyncio.sleep(0)

    assert not readiness.is_ready()
    assert not draining.done()

    await runtime.close_attempt(attempt)
    await draining


@pytest.mark.anyio
async def test_admission_is_bounded_to_configured_gpu_concurrency():
    runtime, _readiness = create_runtime(max_jobs=1)
    await runtime.start()
    active = await runtime.prepare_attempt(envelope())

    with pytest.raises(PodRuntimeBusy, match="pod_capacity_reached"):
        await runtime.prepare_attempt(envelope())

    await runtime.close_attempt(active)


@pytest.mark.anyio
async def test_role_mismatch_is_rejected_without_claiming_backend_attempt():
    runtime, _readiness = create_runtime()
    await runtime.start()

    attempt = await runtime.prepare_attempt(envelope(JobType.PIPELINE))

    assert isinstance(attempt.result, AttemptRejection)
    assert attempt.result.event == "capability_rejected"
