import asyncio
from pathlib import Path

from hear.contracts.jobs import AttemptEnvelope
from hear.contracts.outcomes import ArtifactManifest
from hear.models.discovery import ContentDiscoveryProfile
from hear.workflows.pipeline import PipelineWorkflow


class Audio:
    def __init__(self, root: Path) -> None:
        self._root = root

    async def download_to_wav(self, url, workspace, preserve_channels=True):
        path = workspace.file("source.wav")
        path.write_bytes(b"audio")
        return path


class Transcriber:
    def __init__(self, transcript: str) -> None:
        self._transcript = transcript

    async def transcribe_file(self, path, **kwargs):
        if not self._transcript:
            return {
                "transcript": "",
                "segments": [],
                "language": None,
                "silent": True,
            }
        return {
            "transcript": self._transcript,
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": self._transcript,
                    "words": [],
                }
            ],
            "language": "en",
            "duration": 1.0,
            "silent": False,
        }


class Moderator:
    def __init__(self, flagged: bool) -> None:
        self._flagged = flagged

    async def moderate(self, text):
        return {
            "flagged": self._flagged,
            "severity": "high" if self._flagged else "none",
            "intent": "harmful" if self._flagged else "safe",
            "reason": "",
            "flagged_categories": [],
            "blocked_words_found": [],
        }


class Categorizer:
    def __init__(self) -> None:
        self.calls = 0

    async def categorize(self, **kwargs):
        self.calls += 1
        return {
            "tags": ["#news"],
            "categories": ["News"],
            "sentiment": "neutral",
        }


class Discovery:
    def __init__(self) -> None:
        self.calls = 0

    async def build_profile(self, transcript, **kwargs):
        self.calls += 1
        return ContentDiscoveryProfile(
            content_id=kwargs.get("content_id"),
            main_topic="News",
            one_line_description="News update",
            summary_short="News update",
        )


class Native:
    async def run(self, function, *args, **kwargs):
        return function(*args, **kwargs)


class Workflow(PipelineWorkflow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compressions = 0

    async def _compress(self, envelope, source, workspace):
        self.compressions += 1
        artifact = ArtifactManifest(
            bucket_name="bucket",
            object_key="prefix/source/job.mp3",
            size_bytes=10,
            sha256="a" * 64,
            content_type="audio/mpeg",
            audio_url="https://cdn.example/job.mp3",
        )
        return {
            "audio_url": artifact.audio_url,
            "b2_key": artifact.object_key,
            "bucket_name": artifact.bucket_name,
            "format": "mp3",
            "bitrate_kbps": 96,
            "duration_seconds": 1.0,
            "size_bytes": 10,
            "source_size_bytes": 20,
            "size_reduction_bytes": 10,
            "size_reduction_pct": 50.0,
        }, artifact


def envelope():
    return AttemptEnvelope.model_validate(
        {
            "job_id": "job-1",
            "run_id": "run-1",
            "attempt_id": "attempt-1",
            "job_type": "pipeline",
            "track_id": "track-1",
            "user_id": "user-1",
            "source": {
                "url": "https://media.example/source.mp3",
                "revision": 1,
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
            "options": {},
            "artifact_prefix": "jobs/job-1/attempt-1",
            "deadline": "2026-09-27T00:00:00Z",
            "reporting_grant": "grant",
            "backend_base_url": "https://api.example/api/v1",
        }
    )


def workflow(tmp_path, transcript, flagged):
    categorizer = Categorizer()
    discovery = Discovery()
    instance = Workflow(
        Transcriber(transcript),
        Moderator(flagged),
        categorizer,
        discovery,
        Audio(tmp_path),
        object(),
        Native(),
        workspace_root=tmp_path,
    )
    return instance, categorizer, discovery


def run(instance):
    async def collect():
        return [item async for item in instance.stream(envelope())]

    return asyncio.run(collect())


def test_pipeline_normal_runs_enrichment_and_compression(tmp_path):
    instance, categorizer, discovery = workflow(tmp_path, "Local news update", False)
    events = run(instance)
    outcome = events[-1].data["outcome"]
    assert outcome["status"] == "completed"
    assert categorizer.calls == 1
    assert discovery.calls == 1
    assert instance.compressions == 1
    assert outcome["result"]["compressed_audio"]["format"] == "mp3"


def test_pipeline_flagged_skips_enrichment_but_compresses(tmp_path):
    instance, categorizer, discovery = workflow(tmp_path, "Direct threat", True)
    events = run(instance)
    outcome = events[-1].data["outcome"]
    assert outcome["result"]["moderation"]["flagged"] is True
    assert outcome["result"]["categorization"] is None
    assert categorizer.calls == 0
    assert discovery.calls == 0
    assert instance.compressions == 1


def test_pipeline_no_content_returns_before_compression(tmp_path):
    instance, categorizer, discovery = workflow(tmp_path, "", False)
    events = run(instance)
    outcome = events[-1].data["outcome"]
    assert outcome["result"]["report"]["code"] == "content_not_detected"
    assert outcome["result"]["moderation"]["intent"] == "no_content"
    assert categorizer.calls == 0
    assert discovery.calls == 0
    assert instance.compressions == 0
