from datetime import UTC, datetime, timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hear.api.routers.jobs import JobsRouter
from hear.contracts.events import ExecutionEvent, ExecutionEventType
from hear.contracts.jobs import AttemptClaim, ClaimDecision, JobType
from hear.execution.executor import JobExecutor
from hear.runtime.pod import PodRuntime
from hear.runtime.roles import WorkerRole


class FakeReadiness:
    def __init__(self):
        self.draining = False
        self.checks = {}

    def add_check(self, name, check):
        self.checks[name] = check

    def set_draining(self, value):
        self.draining = value

    def is_ready(self):
        return not self.draining and all(check() for check in self.checks.values())


class FakeBackend:
    def __init__(self, decision=ClaimDecision.EXECUTE):
        self.decision = decision
        self.claims = 0

    async def claim(self, _envelope):
        self.claims += 1
        return AttemptClaim(decision=self.decision)

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


def attempt_envelope():
    return {
        "job_id": "job-1",
        "run_id": "run-1",
        "attempt_id": "attempt-1",
        "job_type": "transcription",
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
            "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        },
        "artifact_prefix": "jobs/job-1/attempt-1",
        "deadline": (datetime.now(UTC) + timedelta(minutes=30)).isoformat(),
        "reporting_grant": "grant",
        "backend_base_url": "https://api.example.com",
    }


def create_client(decision=ClaimDecision.EXECUTE):
    readiness = FakeReadiness()
    backend = FakeBackend(decision)
    runtime = PodRuntime(
        readiness,
        WorkerRole.TRANSCRIPTION,
        JobExecutor({JobType.TRANSCRIPTION: FakeWorkflow()}),
        backend,
        api_key="pod-service-key",
    )
    import asyncio

    asyncio.run(runtime.start())
    app = FastAPI()
    app.include_router(JobsRouter(runtime, "pod-service-key").router)
    return TestClient(app), backend, runtime


class TestPodHttpApi:
    def test_attempt_stream_rejects_invalid_bearer_key(self):
        client, backend, _runtime = create_client()

        response = client.post("/v1/attempts/stream", json=attempt_envelope())

        assert response.status_code == 401
        assert backend.claims == 0

    def test_attempt_stream_returns_canonical_events_as_sse(self):
        client, backend, _runtime = create_client()

        response = client.post(
            "/v1/attempts/stream",
            headers={"Authorization": "Bearer pod-service-key"},
            json=attempt_envelope(),
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert "id: event-1\nevent: progress\ndata:" in response.text
        assert '"progress_pct":20.0' in response.text
        assert backend.claims == 1

    def test_attempt_stream_emits_claim_rejection(self):
        client, backend, _runtime = create_client(ClaimDecision.ALREADY_COMPLETED)

        response = client.post(
            "/v1/attempts/stream",
            headers={"Authorization": "Bearer pod-service-key"},
            json=attempt_envelope(),
        )

        assert response.status_code == 200
        assert "event: claim_rejected" in response.text
        assert '"decision":"already_completed"' in response.text
        assert backend.claims == 1
