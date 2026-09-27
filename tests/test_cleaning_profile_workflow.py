"""Transport/outcome tests; a deliberate fake isolates the workflow from real DSP tests."""

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from hear.contracts.cleaning import MagicCleanProfile
from hear.contracts.jobs import AttemptEnvelope
from hear.contracts.outcomes import ArtifactManifest
from hear.workflows.available_magic_clean import AvailableMagicCleanWorkflow


class InlineNative:
    async def run(self, function, *args, **kwargs):
        return function(*args, **kwargs)


class DownloadFixture:
    async def download_source(self, url, workspace):
        self.workspace = workspace.path
        source = workspace.file("input.audio")
        source.write_bytes(b"pinned-test-input")
        return source


class ReportFixture:
    profile = "natural"
    engine = "deepfilternet3"
    supported_profiles = {profile.value for profile in MagicCleanProfile}

    def __init__(self, invalid=None):
        self.invalid = invalid
        self.called = False

    def clean(self, source, master, workspace, options, deadline, timeout):
        self.called = True
        master.write_bytes(b"fake-master-for-transport-test-only")
        if self.invalid != "missing_delivery":
            (workspace / "delivery_audio.mp3").write_bytes(b"fake-delivery-for-transport-test-only")
        return {
            "engine": self.engine,
            "profile": "wrong_profile" if self.invalid == "profile" else options["profile"],
            "duration_seconds": 1,
            "technical_validation": "failed" if self.invalid == "validation" else "passed",
        }


class StorageFixture:
    def __init__(self):
        self.uploads = []

    def create(self, grant):
        return self

    def key(self, *parts):
        return "/".join(parts)

    def upload_file(self, source, key, sha256, content_type):
        self.uploads.append(key)
        return ArtifactManifest(
            bucket_name="bucket",
            object_key=key,
            size_bytes=source.stat().st_size,
            sha256=sha256,
            content_type=content_type,
        )

    def upload_json(self, value, key):
        self.report = value
        self.uploads.append(key)
        return ArtifactManifest(
            bucket_name="bucket",
            object_key=key,
            size_bytes=1,
            sha256="a" * 64,
            content_type="application/json",
        )


def envelope(profile, digest=None):
    return AttemptEnvelope(
        job_id="job-1",
        run_id="run-1",
        attempt_id="attempt-1",
        job_type="magic_clean",
        track_id="track-1",
        user_id="user-1",
        source={"url": "https://example.com/input.mp3", "revision": 1, "file_sha256": digest},
        storage={
            "endpoint_url": "https://s3.example.com",
            "bucket_name": "bucket",
            "key_id": "key",
            "application_key": "test-secret",
            "folder_prefix": "users/user-1/jobs/",
            "public_base_url": "https://cdn.example.com",
            "expires_at": datetime.now(UTC) + timedelta(hours=1),
        },
        options={"profile": profile},
        artifact_prefix="jobs/job-1/attempt-1",
        deadline=datetime.now(UTC) + timedelta(minutes=5),
        reporting_grant="test-grant",
        backend_base_url="https://api.example.com",
    )


def run(tmp_path, profile="natural", digest=None, invalid=None):
    audio, storage, cleaner = DownloadFixture(), StorageFixture(), ReportFixture(invalid)
    workflow = AvailableMagicCleanWorkflow(
        audio,
        storage,
        InlineNative(),
        workspace_root=tmp_path,
        timeout_seconds=60,
        model_cleaner=cleaner,
    )

    async def collect():
        return [event async for event in workflow.stream(envelope(profile, digest))]

    events = asyncio.run(collect())
    assert not audio.workspace.exists()
    return SimpleNamespace(
        events=events, outcome=events[-1].data["outcome"], storage=storage, cleaner=cleaner
    )


@pytest.mark.parametrize("profile", [p.value for p in MagicCleanProfile])
def test_all_profiles_publish_three_artifacts_with_approval(tmp_path, profile):
    digest = hashlib.sha256(b"pinned-test-input").hexdigest()
    result = run(tmp_path, profile, digest)
    assert result.outcome["status"] == "completed"
    assert result.outcome["result"]["profile"] == profile
    assert result.outcome["result"]["requires_approval"] is True
    assert len(result.storage.uploads) == 3
    assert result.storage.report["source_sha256"] == digest
    assert result.storage.report["source_revision"] == 1


def test_changed_source_fails_before_inference_or_upload(tmp_path):
    result = run(tmp_path, digest="a" * 64)
    assert result.outcome["status"] == "failed"
    assert result.outcome["error_code"] == "source_mismatch"
    assert not result.cleaner.called and not result.storage.uploads


@pytest.mark.parametrize("invalid", ["missing_delivery", "validation", "profile"])
def test_unvalidated_or_mismatched_output_is_never_published(tmp_path, invalid):
    result = run(tmp_path, invalid=invalid)
    assert result.outcome["status"] == "failed"
    assert result.outcome["error_code"] == "invalid_audio"
    assert not result.storage.uploads
