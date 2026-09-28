"""Real codec/timeline tests with an explicit synthetic model double, not a Fish quality benchmark."""

import asyncio
import hashlib
import io
import json
import shutil
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from hear.bootstrap import RuntimeBootstrap
from hear.contracts.jobs import AttemptClaim, AttemptEnvelope, ClaimDecision, JobType
from hear.contracts.outcomes import ArtifactManifest
from hear.contracts.reconstruction import ReconstructionOptions
from hear.execution.executor import JobExecutor
from hear.execution.native import NativeExecutor
from hear.inference.fish_speech import FishSpeechEngine
from hear.queue.topology import RabbitMQTopology
from hear.runtime.roles import WorkerRole
from hear.runtime.serverless import ServerlessRuntime
from hear.services.reconstruction.fish_renderer import FishReconstructionRenderer
from hear.workflows.fish_reconstruction import FishReconstructionWorkflow


@pytest.mark.parametrize(
    "raw",
    [
        {
            "changes": [
                {
                    "segment_start": 1,
                    "segment_end": 2,
                    "replacement_audio_url": "https://example.com/audio.wav",
                }
            ]
        },
        {"changes": [{"segment_start": 1, "segment_end": 2, "new_text": ""}]},
        {
            "same_speaker": "false",
            "changes": [{"segment_start": 1, "segment_end": 2, "new_text": "New"}],
        },
        {
            "same_speaker": True,
            "changes": [{"segment_start": 1, "segment_end": 2, "new_text": "New"}],
        },
        {
            "same_speaker": False,
            "changes": [{"segment_start": float("nan"), "segment_end": 2, "new_text": "New"}],
        },
        {
            "same_speaker": False,
            "changes": [
                {"segment_start": 1, "segment_end": 3, "new_text": "One"},
                {"segment_start": 2, "segment_end": 4, "new_text": "Two"},
            ],
        },
        {
            "same_speaker": False,
            "changes": [{"segment_start": 1, "segment_end": 2, "new_text": "x" * 2001}],
        },
    ],
)
def test_invalid_or_non_tts_requests_rejected(raw):
    with pytest.raises(ValueError):
        ReconstructionOptions.validate_operation("replace_segments", raw)


def test_rebuild_requires_matching_voice_reference():
    with pytest.raises(ValueError, match="reference"):
        ReconstructionOptions.validate_operation(
            "rebuild", {"edited_transcript": "Updated narration"}
        )
    result = ReconstructionOptions.validate_operation(
        "rebuild",
        {
            "edited_transcript": "Updated narration",
            "reference": {
                "start_seconds": 0,
                "end_seconds": 5,
                "text": "The actual words in those five seconds.",
            },
        },
    )
    assert result.same_speaker


def test_cleaner_mode_never_changes_reconstruction_engine():
    for mode in ("available", "certified"):
        bootstrap = RuntimeBootstrap({"HEAR_OPTIONAL_ENGINE_MODE": mode})
        assert not bootstrap._uses_available_engine(WorkerRole.RECONSTRUCTION)
        snapshot = bootstrap.readiness(WorkerRole.RECONSTRUCTION).snapshot()
        assert "fish-speech-s2-pro:permission_required" in snapshot["license_blockers"]


def test_fish_uses_different_queue_from_legacy_splicing():
    queue = RabbitMQTopology().binding(WorkerRole.RECONSTRUCTION)
    assert queue.routing_key == "reconstruction.fish_tts.v4"
    assert queue.queue != "hear.ai.reconstruction.v2"


def test_no_global_voice_cache_ids():
    with pytest.raises(ValueError, match="global_voice"):
        FishSpeechEngine.validate_request("Text", None, "other-workspace", "en", 1024)


def test_text_chunking_retains_every_word():
    text = "This is a precise replacement sentence. " * 100
    chunks = FishReconstructionRenderer.text_chunks(text)
    assert " ".join(chunks).split() == text.split()
    assert max(map(len, chunks)) <= 400


class SyntheticSpeechDouble:
    def __init__(self):
        self.calls = []

    async def generate_speech(self, **kwargs):
        self.calls.append(kwargs)
        t = np.arange(24000) / 48000
        wave = (0.06 * np.sin(2 * np.pi * 330 * t)).astype("float32")
        buffer = io.BytesIO()
        sf.write(buffer, wave, 48000, format="WAV", subtype="FLOAT")
        return buffer.getvalue()


class FileSource:
    def __init__(self, path):
        self.path = path
        self.downloaded = False

    async def download_source(self, url, workspace):
        self.downloaded = True
        target = workspace.file("source.wav")
        shutil.copyfile(self.path, target)
        return target


class MemoryStorage:
    bucket_name = "test-bucket"

    def __init__(self):
        self.uploads = {}
        self.manifest = None

    def create(self, context):
        return self

    def key(self, *parts):
        return "creators/owner/audio/" + "/".join(parts)

    def upload_file(self, path, key, *, sha256, content_type):
        data = Path(path).read_bytes()
        assert hashlib.sha256(data).hexdigest() == sha256
        self.uploads[key] = data
        return ArtifactManifest(
            bucket_name=self.bucket_name,
            object_key=key,
            size_bytes=len(data),
            sha256=sha256,
            content_type=content_type,
            audio_url="https://cdn.example.com/" + key,
        )

    def upload_json(self, payload, key):
        self.manifest = payload
        data = json.dumps(payload).encode()
        return ArtifactManifest(
            bucket_name=self.bucket_name,
            object_key=key,
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            content_type="application/json",
        )


def source_file(tmp_path, channels=1):
    t = np.arange(48000 * 4) / 48000
    wave = (0.05 * np.sin(2 * np.pi * 210 * t)).astype("float32")
    values = wave if channels == 1 else np.column_stack([wave, -wave])
    path = tmp_path / "original.wav"
    sf.write(path, values, 48000, subtype="FLOAT")
    return path, values


def request(source, operation="replace_segments", options=None):
    return AttemptEnvelope(
        job_id="job",
        run_id="run",
        attempt_id="attempt",
        job_type="reconstruction",
        operation=operation,
        track_id="track",
        user_id="user",
        backend_id="backend-a",
        source={
            "url": "https://cdn.example.com/source.wav",
            "revision": 7,
            "file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
        storage={
            "endpoint_url": "https://s3.example.com",
            "bucket_name": "test-bucket",
            "key_id": "not-a-real-key",
            "application_key": "test-only",
            "folder_prefix": "creators/owner/audio/jobs/job/",
            "public_base_url": "https://cdn.example.com",
            "expires_at": datetime.now(UTC) + timedelta(hours=1),
        },
        options=options
        or {
            "same_speaker": True,
            "changes": [
                {
                    "segment_start": 1,
                    "segment_end": 2,
                    "new_text": "Replaced words.",
                    "original_text": "Original words.",
                }
            ],
        },
        artifact_prefix="creators/owner/audio/jobs/job/attempt",
        deadline=datetime.now(UTC) + timedelta(minutes=5),
        reporting_grant="test-only",
        backend_base_url="https://api.example.com",
    )


@pytest.mark.parametrize("channels", [1, 2])
def test_full_tts_workflow_returns_audio_ownership_and_changed_timeline(tmp_path, channels):
    path, values = source_file(tmp_path, channels)
    old_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    fish = SyntheticSpeechDouble()
    storage = MemoryStorage()
    audio = FileSource(path)

    async def run():
        native = NativeExecutor("fish-test")
        try:
            workflow = FishReconstructionWorkflow(
                FishReconstructionRenderer(fish, native),
                audio,
                storage,
                native,
                workspace_root=tmp_path / "jobs",
            )
            return [event async for event in workflow.stream(request(path))]
        finally:
            await native.close()

    events = asyncio.run(run())
    outcome = events[-1].data["outcome"]
    assert outcome["status"] == "completed"
    result = outcome["result"]["reconstructed_audio"]
    assert result["engine"] == "fish_speech_s2_pro" and result["backend_id"] == "backend-a"
    assert result["source_revision"] == 7 and result["requires_approval"] is True
    assert result["duration"] == 3.5 and result["output_frames"] == 168000
    assert result["segments"][0]["output_start_frame"] == 48000
    assert result["segments"][0]["output_end_frame"] == 72000
    assert result["segments"][0]["audio_url"]
    assert result["channels"] == channels and result["delivery_measurement"]["true_peak_dbtp"] <= -1
    assert (
        fish.calls[0]["text"] == "Replaced words."
        and fish.calls[0]["references"][0]["text"] == "Original words."
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == old_digest
    assert len(storage.uploads) == 3 and result["word_accuracy_verified"] is False


def test_source_revision_hash_failure_prevents_tts_and_upload(tmp_path):
    path, _ = source_file(tmp_path)
    fish = SyntheticSpeechDouble()
    storage = MemoryStorage()
    value = request(path)
    value = value.model_copy(
        update={"source": value.source.model_copy(update={"file_sha256": "a" * 64})}
    )

    async def run():
        native = NativeExecutor("bad-source")
        try:
            workflow = FishReconstructionWorkflow(
                FishReconstructionRenderer(fish, native),
                FileSource(path),
                storage,
                native,
                workspace_root=tmp_path / "jobs",
            )
            return [event async for event in workflow.stream(value)]
        finally:
            await native.close()

    with pytest.raises(Exception, match="digest_mismatch"):
        asyncio.run(run())
    assert not fish.calls and not storage.uploads


class BackendRecorder:
    def __init__(self):
        self.outcomes = []
        self.events = []

    async def claim(self, envelope):
        return AttemptClaim(decision=ClaimDecision.EXECUTE)

    async def heartbeat(self, *args):
        pass

    async def event(self, envelope, value):
        self.events.append(value)

    async def outcome(self, envelope, value):
        self.outcomes.append(value)


def test_serverless_posts_authoritative_outcome_before_returning_completion(tmp_path):
    path, _ = source_file(tmp_path)
    fish = SyntheticSpeechDouble()
    storage = MemoryStorage()
    backend = BackendRecorder()
    provider = SimpleNamespace(serverless=SimpleNamespace(progress_update=lambda *args: None))

    async def run():
        native = NativeExecutor("serverless-fish")
        try:
            workflow = FishReconstructionWorkflow(
                FishReconstructionRenderer(fish, native),
                FileSource(path),
                storage,
                native,
                workspace_root=tmp_path / "jobs",
            )
            runtime = ServerlessRuntime(
                WorkerRole.RECONSTRUCTION,
                JobExecutor({JobType.RECONSTRUCTION: workflow}),
                backend,
                provider,
            )
            items = []
            async for item in runtime.handler(
                {"id": "provider-job", "input": request(path).model_dump(mode="json")}
            ):
                if item.get("event") == "outcome":
                    assert len(backend.outcomes) == 1
                items.append(item)
            return items
        finally:
            await native.close()

    assert asyncio.run(run())[-1]["data"]["outcome"]["result"]["engine"] == "fish_speech_s2_pro"
    assert len(backend.outcomes) == 1 and backend.events


def test_fish_serverless_does_not_advertise_unmeasured_parallel_tts():
    with pytest.raises(ValueError, match="one_job_per_worker"):
        ServerlessRuntime(
            WorkerRole.RECONSTRUCTION, JobExecutor({}), BackendRecorder(), max_concurrent_jobs=4
        )


def test_cancelled_fish_process_is_terminated_and_unhealthy():
    engine = FishSpeechEngine.__new__(FishSpeechEngine)
    state = {"alive": True, "terminated": False}

    def terminate():
        state.update(alive=False, terminated=True)

    engine._process = SimpleNamespace(
        is_alive=lambda: state["alive"], terminate=terminate, join=lambda **kw: None, kill=terminate
    )
    engine._connection = SimpleNamespace(close=lambda: None)
    engine._faulted = False
    engine._closed = False
    cancellation = threading.Event()
    cancellation.set()
    with pytest.raises(RuntimeError, match="cancelled"):
        engine._receive(10, cancellation)
    assert state["terminated"]
    with pytest.raises(RuntimeError, match="unavailable"):
        engine.check_health()


@pytest.mark.parametrize(
    "operation,options,expected,calls",
    [
        ("remove_segments", {"segment_start": 1, "segment_end": 2}, 3.0, 0),
        (
            "rebuild",
            {"same_speaker": False, "edited_transcript": "Completely new narration."},
            0.5,
            1,
        ),
        (
            "replace_segments",
            {
                "same_speaker": False,
                "changes": [{"segment_start": 2, "segment_end": 2, "new_text": "An insertion."}],
            },
            4.5,
            1,
        ),
        (
            "edit_transcript",
            {
                "same_speaker": False,
                "changes": [
                    {"segment_start": 1, "segment_end": 2, "new_text": "First."},
                    {"segment_start": 3, "segment_end": 3.5, "new_text": "Second."},
                ],
            },
            3.5,
            2,
        ),
    ],
)
def test_all_edit_operations_have_correct_output_duration(
    tmp_path, operation, options, expected, calls
):
    path, _ = source_file(tmp_path)
    fish = SyntheticSpeechDouble()
    storage = MemoryStorage()

    async def run():
        native = NativeExecutor("fish-operations")
        try:
            workflow = FishReconstructionWorkflow(
                FishReconstructionRenderer(fish, native),
                FileSource(path),
                storage,
                native,
                workspace_root=tmp_path / "jobs",
            )
            events = [e async for e in workflow.stream(request(path, operation, options))]
            return events[-1].data["outcome"]["result"]["reconstructed_audio"]
        finally:
            await native.close()

    result = asyncio.run(run())
    assert result["duration"] == expected and len(fish.calls) == calls


def test_transcription_only_prefers_its_own_lane(tmp_path):
    path, _ = source_file(tmp_path)
    value = request(path).model_copy(
        update={"job_type": JobType.TRANSCRIPTION, "operation": None, "options": {}}
    )
    assert (
        RabbitMQTopology().role_for(value, {WorkerRole.PIPELINE, WorkerRole.TRANSCRIPTION})
        == WorkerRole.TRANSCRIPTION
    )
