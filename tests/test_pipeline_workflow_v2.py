from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hear.contracts.events import ExecutionEventType
from hear.contracts.jobs import AttemptEnvelope, JobType
from hear.contracts.outcomes import ArtifactManifest
from hear.execution.native import NativeExecutor
from hear.workflows.pipeline import PipelineWorkflow


class FakeAudio:
    async def download_to_wav(self, url, workspace, preserve_channels=True):
        path = workspace.file("source.wav")
        path.write_bytes(b"wav")
        return path


class FakeTranscriber:
    async def transcribe_file(self, path, **kwargs):
        await kwargs["progress"].publish(50)
        return {
            "transcript": "Local community news",
            "segments": [{"text": "Local community news"}],
            "language": "en",
            "duration": 2.0,
            "audio_duration": 2.0,
            "confidence": 0.9,
            "silent": False,
        }


class FakeModerator:
    async def moderate(self, text):
        return {
            "flagged": False,
            "severity": "none",
            "intent": "safe",
            "reason": "",
            "flagged_categories": [],
            "blocked_words_found": [],
        }


class FakeCategorizer:
    async def categorize(self, **kwargs):
        return {
            "tags": ["#news"],
            "categories": ["News"],
            "confidence_scores": {},
            "sentiment": "neutral",
        }


class FakeDiscovery:
    async def build_profile(self, *args, **kwargs):
        return None


class FakeStorage:
    bucket_name = "bucket"

    def key(self, *parts):
        return "users/user-1/" + "/".join(parts)

    def upload_file(self, path, key, *, sha256, content_type=None):
        return ArtifactManifest(
            bucket_name="bucket",
            object_key=key,
            size_bytes=3,
            sha256=sha256,
            content_type=content_type or "audio/mpeg",
            audio_url="https://cdn.example/source.mp3",
        )

    def upload_json(self, payload, key):
        return ArtifactManifest(
            bucket_name="bucket",
            object_key=key,
            size_bytes=100,
            sha256="a" * 64,
            content_type="application/json",
            audio_url="https://cdn.example/pipeline.json",
        )


class FakeStorageFactory:
    def create(self, context):
        return FakeStorage()


def envelope():
    return AttemptEnvelope(
        job_id="job-1",
        run_id="run-1",
        attempt_id="attempt-1",
        job_type=JobType.PIPELINE,
        track_id="track-1",
        user_id="user-1",
        source={"url": "https://example.com/a.mp3", "revision": 1},
        storage={
            "endpoint_url": "https://s3.example.com",
            "bucket_name": "bucket",
            "key_id": "key",
            "application_key": "secret",
            "folder_prefix": "users/user-1/",
            "public_base_url": "https://cdn.example/media",
            "expires_at": datetime.now(UTC) + timedelta(hours=1),
        },
        artifact_prefix="users/user-1/jobs/job-1/attempt-1",
        deadline=datetime.now(UTC) + timedelta(minutes=30),
        reporting_grant="grant",
        backend_base_url="https://api.example.com",
    )


@pytest.mark.anyio
async def test_pipeline_streams_all_core_stages(tmp_path: Path):
    native = NativeExecutor("pipeline-test")
    workflow = PipelineWorkflow(
        FakeTranscriber(),
        FakeModerator(),
        FakeCategorizer(),
        FakeDiscovery(),
        FakeAudio(),
        FakeStorageFactory(),
        native,
        workspace_root=tmp_path,
    )

    async def fake_encode(source, attempt, workspace):
        target = workspace.file("source.mp3")
        target.write_bytes(b"mp3")
        return target

    workflow._encode = fake_encode
    events = [event async for event in workflow.stream(envelope())]
    await native.close()
    stages = [event.stage for event in events]
    assert "transcribing" in stages
    assert "moderating" in stages
    assert "categorizing" in stages
    assert "discovering" in stages
    assert "compressing" in stages
    assert events[-1].event == ExecutionEventType.OUTCOME
    outcome = events[-1].data["outcome"]
    assert outcome["status"] == "completed"
    assert outcome["result"]["pipeline_manifest"]["object_key"].endswith("pipeline.json")
