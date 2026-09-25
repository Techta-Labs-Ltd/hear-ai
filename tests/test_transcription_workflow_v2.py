from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hear.contracts.events import ExecutionEventType
from hear.contracts.jobs import AttemptEnvelope, JobType
from hear.contracts.outcomes import ArtifactManifest
from hear.execution.native import NativeExecutor
from hear.workflows.transcription import TranscriptionWorkflow


class FakeAudio:
    async def download_to_wav(self, url, workspace, preserve_channels=True):
        path = workspace.file("source.wav")
        path.write_bytes(b"wav")
        return path


class FakeTranscriber:
    async def transcribe_file(self, path, **kwargs):
        progress = kwargs["progress"]
        await progress.publish(25)
        await progress.publish(75)
        return {
            "transcript": "hello world",
            "segments": [],
            "language": "en",
            "duration": 1.0,
            "confidence": 0.9,
            "silent": False,
        }


class FakeStorage:
    def key(self, *parts):
        return "users/user-1/" + "/".join(parts)

    def upload_json(self, payload, key):
        return ArtifactManifest(
            bucket_name="bucket",
            object_key=key,
            size_bytes=100,
            sha256="a" * 64,
            content_type="application/json",
            audio_url="https://cdn.example.com/transcription.json",
        )


class FakeStorageFactory:
    def create(self, context):
        return FakeStorage()


class TestTranscriptionWorkflow:
    @pytest.mark.anyio
    async def test_streams_progress_and_compact_outcome(self, tmp_path: Path):
        native = NativeExecutor("test")
        workflow = TranscriptionWorkflow(
            FakeTranscriber(),
            FakeAudio(),
            FakeStorageFactory(),
            native,
            workspace_root=tmp_path,
        )
        envelope = AttemptEnvelope(
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
                "folder_prefix": "users/user-1/",
                "public_base_url": "https://cdn.example.com/media",
                "expires_at": datetime.now(UTC) + timedelta(hours=1),
            },
            artifact_prefix="users/user-1/jobs/job-1/attempt-1",
            deadline=datetime.now(UTC) + timedelta(minutes=30),
            reporting_grant="grant",
            backend_base_url="https://api.example.com",
        )
        events = [event async for event in workflow.stream(envelope)]
        await native.close()
        assert events[0].event == ExecutionEventType.STARTED
        assert any(event.event == ExecutionEventType.PROGRESS for event in events)
        assert events[-1].event == ExecutionEventType.OUTCOME
        outcome = events[-1].data["outcome"]
        assert outcome["status"] == "completed"
        assert "transcript" not in outcome["result"]
        assert outcome["result"]["transcription_manifest"]["object_key"].endswith(
            "transcription.json"
        )