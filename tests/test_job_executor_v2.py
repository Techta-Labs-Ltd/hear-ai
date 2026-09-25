from datetime import UTC, datetime, timedelta

import pytest

from hear.contracts.events import ExecutionEvent, ExecutionEventType
from hear.contracts.jobs import AttemptEnvelope, JobType
from hear.execution.executor import JobExecutor


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
            progress_pct=50,
        )


class TestJobExecutor:
    @staticmethod
    def envelope() -> AttemptEnvelope:
        return AttemptEnvelope(
            job_id="job-1",
            run_id="run-1",
            attempt_id="attempt-1",
            job_type=JobType.TRANSCRIPTION,
            track_id="track-1",
            user_id="user-1",
            source={"url": "https://example.com/a.mp3", "revision": 1},
            storage={
                "endpoint_url": "https://s3.example.com",
                "bucket_name": "bucket",
                "key_id": "key",
                "application_key": "secret",
                "folder_prefix": "users/user-1/jobs/",
                "public_base_url": "https://cdn.example.com/media",
                "expires_at": datetime.now(UTC) + timedelta(hours=1),
            },
            artifact_prefix="jobs/job-1/attempt-1",
            deadline=datetime.now(UTC) + timedelta(minutes=30),
            reporting_grant="grant",
            backend_base_url="https://api.example.com",
        )

    @pytest.mark.anyio
    async def test_stream_delegates_to_injected_workflow(self):
        executor = JobExecutor({JobType.TRANSCRIPTION: FakeWorkflow()})
        events = [event async for event in executor.stream(self.envelope())]
        assert len(events) == 1
        assert events[0].progress_pct == 50