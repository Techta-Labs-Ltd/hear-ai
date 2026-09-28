import asyncio
from dataclasses import dataclass

from hear.contracts.jobs import AttemptEnvelope
from hear.workflows.reconstruction import ReconstructionWorkflow


class Audio:
    async def download_to_wav(self, url, workspace, preserve_channels=True):
        path = workspace.file("source.wav")
        path.write_bytes(b"audio")
        return path


class Storage:
    bucket_name = "bucket"

    def key(self, *parts):
        return "prefix/" + "/".join(parts)

    def upload_file(self, local_path, object_key, *, sha256, content_type=None):
        raise AssertionError("fake synthesizer does not upload")


class StorageFactory:
    def create(self, context):
        return Storage()


@dataclass
class Segment:
    segment_start: float
    segment_end: float
    b2_key: str
    audio_url: str
    duration: float
    is_deletion: bool
    bucket_name: str


@dataclass
class Result:
    b2_key: str
    audio_url: str
    duration: float
    segments: list
    bucket_name: str


class Synthesizer:
    def __init__(self):
        self.calls = []

    async def reconstruct_segments(self, source, track_id, changes, storage, **kwargs):
        self.calls.append(("segments", source, track_id, changes, kwargs))
        return Result(
            "prefix/reconstructed/job.mp3",
            "https://cdn.example/job.mp3",
            8.0,
            [
                Segment(
                    1.0,
                    2.0,
                    "prefix/segments/one.mp3",
                    "https://cdn.example/one.mp3",
                    1.1,
                    False,
                    "bucket",
                )
            ],
            "bucket",
        )

    async def rebuild_track_audio(self, source, edited, track_id, job_id, storage, **kwargs):
        self.calls.append(("rebuild", source, edited, kwargs))
        return Result(
            "prefix/reconstructed/job.mp3",
            "https://cdn.example/job.mp3",
            9.0,
            [],
            "bucket",
        )

    async def remove_segment(self, source, track_id, start, end, storage, job_id, *, workspace=None):
        self.calls.append(("remove", start, end))
        return Result(
            "prefix/reconstructed/job.mp3",
            "https://cdn.example/job.mp3",
            7.0,
            [],
            "bucket",
        )


def envelope(operation, options):
    return AttemptEnvelope.model_validate(
        {
            "job_id": "job-1",
            "run_id": "run-1",
            "attempt_id": "attempt-1",
            "job_type": "reconstruction",
            "operation": operation,
            "track_id": "track-1",
            "user_id": "user-1",
            "source": {
                "url": "https://media.example/source.mp3",
                "revision": 2,
            },
            "storage": {
                "endpoint_url": "https://s3.example",
                "bucket_name": "bucket",
                "key_id": "key",
                "application_key": "secret",
                "folder_prefix": "prefix/",
                "public_base_url": "https://cdn.example",
                "expires_at": "2026-09-27T00:00:00Z",
            },
            "options": options,
            "artifact_prefix": "jobs/job-1/attempt-1",
            "deadline": "2026-09-27T00:00:00Z",
            "reporting_grant": "grant",
            "backend_base_url": "https://api.example/api/v1",
        }
    )


def collect(workflow, value):
    async def run():
        return [event async for event in workflow.stream(value)]

    return asyncio.run(run())


def test_reconstruction_segments_are_compute_only_and_require_backend_approval(tmp_path):
    synthesizer = Synthesizer()
    workflow = ReconstructionWorkflow(
        synthesizer,
        Audio(),
        StorageFactory(),
        workspace_root=tmp_path,
    )
    events = collect(
        workflow,
        envelope(
            "replace_segments",
            {
                "same_speaker": True,
                "changes": [
                    {
                        "segment_start": 1.0,
                        "segment_end": 2.0,
                        "new_text": "Updated words",
                        "original_text": "Old words",
                    }
                ],
            },
        ),
    )
    outcome = events[-1].data["outcome"]
    assert outcome["status"] == "completed"
    assert outcome["result"]["requires_approval"] is True
    assert outcome["result"]["operation"] == "replace_segments"
    assert synthesizer.calls[0][0] == "segments"


def test_reconstruction_rebuild_uses_existing_transcript_context(tmp_path):
    synthesizer = Synthesizer()
    workflow = ReconstructionWorkflow(
        synthesizer,
        Audio(),
        StorageFactory(),
        workspace_root=tmp_path,
    )
    events = collect(
        workflow,
        envelope(
            "rebuild",
            {
                "same_speaker": False,
                "edited_transcript": "New complete transcript",
                "original_transcript": "Original complete transcript",
            },
        ),
    )
    assert events[-1].data["outcome"]["result"]["operation"] == "rebuild"
    assert synthesizer.calls[0][0] == "rebuild"


def test_reconstruction_remove_segment_has_no_tts_requirement(tmp_path):
    synthesizer = Synthesizer()
    workflow = ReconstructionWorkflow(
        synthesizer,
        Audio(),
        StorageFactory(),
        workspace_root=tmp_path,
    )
    events = collect(
        workflow,
        envelope(
            "remove_segments",
            {"segment_start": 2.0, "segment_end": 3.0},
        ),
    )
    assert events[-1].data["outcome"]["result"]["operation"] == "remove_segments"
    assert synthesizer.calls[0] == ("remove", 2.0, 3.0)
