from datetime import UTC, datetime, timedelta

import pytest

from hear.contracts.events import ExecutionEvent, ExecutionEventType
from hear.contracts.jobs import ClaimDecision, JobType
from hear.execution.executor import JobExecutor
from hear.runtime.roles import WorkerRole
from hear.runtime.serverless import ServerlessRuntime


class FakeBackend:
    async def claim(self, envelope):
        return ClaimDecision.EXECUTE


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


class TestServerlessRuntime:
    @pytest.mark.anyio
    async def test_handler_yields_canonical_event(self, monkeypatch):
        updates = []
        monkeypatch.setattr(
            "hear.runtime.serverless.runpod.serverless.progress_update",
            lambda job, value: updates.append(value),
        )
        runtime = ServerlessRuntime(
            WorkerRole.TRANSCRIPTION,
            JobExecutor({JobType.TRANSCRIPTION: FakeWorkflow()}),
            FakeBackend(),
        )
        job = {
            "input": {
                "job_id": "job-1",
                "run_id": "run-1",
                "attempt_id": "attempt-1",
                "job_type": "transcription",
                "track_id": "track-1",
                "user_id": "user-1",
                "source": {"url": "https://example.com/a.mp3", "revision": 1},
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
        }
        events = [event async for event in runtime.handler(job)]
        assert events[0]["event"] == "progress"
        assert updates == ["transcribing:20.0"]