import asyncio
import hashlib
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hear.config import settings
from hear.core.backend_registry import BackendRegistry
from hear.core.storage import StorageCredentialsExpiringError
from hear.models.database import AiJob
from hear.models.schemas import PipelineRequest as RequestModel
from hear.models.schemas import SegmentChange
from hear.services.jobs.submission import (
    JobSubmissionService,
    SubmissionConflictError,
    SubmissionPolicy,
)

BACKEND_ID = "backend-a"
SERVICE_KEY = "backend-secret"
STORAGE = {
    "endpoint_url": "https://s3.example.test",
    "bucket_name": "backend-a-bucket",
    "key_id": "key-id",
    "application_key": "application-key",
    "folder_prefix": "users/user/jobs/job",
    "public_base_url": "https://cdn.example.test",
    "expires_at": (datetime.now(UTC) + timedelta(days=2)).isoformat(),
}
REGISTRY_JSON = json.dumps(
    {
        BACKEND_ID: {
            "service_key_sha256": hashlib.sha256(SERVICE_KEY.encode()).hexdigest(),
            "allowed_endpoint_urls": [STORAGE["endpoint_url"]],
            "allowed_buckets": [STORAGE["bucket_name"]],
            "allowed_public_base_urls": [STORAGE["public_base_url"]],
        }
    }
)


@pytest.fixture(autouse=True)
def configured_backend_registry(monkeypatch):
    monkeypatch.setattr(settings, "BACKEND_REGISTRY_JSON", REGISTRY_JSON)
    BackendRegistry.backend_registry.cache_clear()
    yield
    BackendRegistry.backend_registry.cache_clear()


def pipeline_request(**kwargs):
    kwargs.setdefault("backend_id", BACKEND_ID)
    kwargs.setdefault("storage", STORAGE)
    return RequestModel(**kwargs)


class _FakeScalarResult:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


class _FakeSubmissionQuery:
    def __init__(self, db):
        self._db = db

    def filter(self, *_args):
        return self

    def populate_existing(self):
        self._db.populated_existing = True
        return self

    def with_for_update(self):
        self._db.locked_for_update = True
        return self

    def first(self):
        self._db.operations.append("query")
        return self._db.job


class _FakeSubmissionDB:
    def __init__(self, job, *, inserted=False):
        self.job = job
        self.inserted = inserted
        self.operations = []
        self.commits = 0
        self.rolled_back = False
        self.closed = False
        self.populated_existing = False
        self.locked_for_update = False

    def query(self, *_args):
        return _FakeSubmissionQuery(self)

    def execute(self, _statement):
        self.operations.append("execute")
        return _FakeScalarResult("job" if self.inserted else None)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


def _terminal_magic_clean_job(
    request: RequestModel, *, status: str = "cancelled", tombstone: dict | None = None
):
    payload = SubmissionPolicy.normalize_request(request, enforce_magic_clean_storage_ttl=False)
    options = SubmissionPolicy._job_options(payload)
    if tombstone is not None:
        options["magic_clean_cleanup_tombstone"] = tombstone
    return SimpleNamespace(
        id=payload["job_id"],
        backend_id=payload["backend_id"],
        run_id="existing-run",
        track_id=payload["track_id"],
        job_type="magic_clean",
        status=status,
        request_hash=SubmissionPolicy.request_fingerprint(payload),
        job_options=options,
        input_url=payload["audio_url"],
        max_tags=payload["max_tags"],
        edited_transcript=None,
        custom_tags=None,
        storage_context_encrypted="encrypted-original-context",
        error="terminal-error",
    )


def test_job_id_is_not_part_of_payload_fingerprint():
    first = SubmissionPolicy.normalize_request(
        pipeline_request(
            job_id="one", track_id="track", user_id="user", audio_url="https://audio.test/a.mp3"
        )
    )
    second = SubmissionPolicy.normalize_request(
        pipeline_request(
            job_id="two", track_id="track", user_id="user", audio_url="https://audio.test/a.mp3"
        )
    )
    assert SubmissionPolicy.request_fingerprint(first) == SubmissionPolicy.request_fingerprint(
        second
    )


def test_payload_change_changes_fingerprint():
    first = SubmissionPolicy.normalize_request(
        pipeline_request(
            job_id="job", track_id="track", user_id="user", audio_url="https://audio.test/a.mp3"
        )
    )
    second = SubmissionPolicy.normalize_request(
        pipeline_request(
            job_id="job",
            track_id="track",
            user_id="user",
            job_type="magic-clean",
            audio_url="https://audio.test/a.mp3",
        )
    )
    assert first["job_type"] == "pipeline"
    assert second["job_type"] == "magic_clean"
    assert SubmissionPolicy.request_fingerprint(first) != SubmissionPolicy.request_fingerprint(
        second
    )


def test_discovery_is_a_supported_standalone_job():
    request = SubmissionPolicy.normalize_request(
        pipeline_request(
            job_id="job",
            track_id="track",
            user_id="user",
            job_type="discovery",
            audio_url="https://audio.test/a.mp3",
        )
    )
    assert request["job_type"] == "discovery"


def test_legacy_tagging_alias_normalizes_to_categorization():
    request = SubmissionPolicy.normalize_request(
        pipeline_request(
            job_id="job",
            track_id="track",
            user_id="user",
            job_type="tagging",
            edited_transcript="short spoken description",
        )
    )
    assert request["job_type"] == "categorization"


@pytest.mark.parametrize(
    "job_type", ["pipeline", "magic_clean", "transcription", "audio_tag", "discovery"]
)
def test_audio_jobs_require_explicit_audio_url(job_type):
    with pytest.raises(ValueError, match="audio_url is required"):
        SubmissionPolicy.normalize_request(
            pipeline_request(job_id="job", track_id="track", user_id="user", job_type=job_type)
        )


def test_magic_clean_stem_levels_are_normalized_and_fingerprinted():
    default = SubmissionPolicy.normalize_request(
        pipeline_request(
            job_id="job",
            track_id="track",
            user_id="user",
            job_type="magic_clean",
            audio_url="https://audio.test/a.mp3",
        )
    )
    customized = SubmissionPolicy.normalize_request(
        pipeline_request(
            job_id="job",
            track_id="track",
            user_id="user",
            job_type="magic_clean",
            audio_url="https://audio.test/a.mp3",
            speech=50,
            music=10,
            background=10,
        )
    )
    assert default["speech"] == 100
    assert default["music"] == 10
    assert default["background"] == 10
    assert customized["speech"] == 50
    assert customized["music"] == 10
    assert customized["background"] == 10
    assert SubmissionPolicy.request_fingerprint(customized) != SubmissionPolicy.request_fingerprint(
        default
    )


def test_magic_clean_requires_cleanup_safe_storage_credential_lifetime():
    short_lived_storage = {
        **STORAGE,
        "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
    }
    ordinary = SubmissionPolicy.normalize_request(
        pipeline_request(
            job_id="ordinary",
            track_id="track",
            user_id="user",
            audio_url="https://audio.test/a.mp3",
            storage=short_lived_storage,
        )
    )
    assert ordinary["job_type"] == "pipeline"
    required_ttl = settings.MAGIC_CLEAN_STORAGE_CREDENTIAL_MIN_TTL_SECONDS
    with pytest.raises(
        StorageCredentialsExpiringError, match=f"remain valid for at least {required_ttl:g} seconds"
    ):
        SubmissionPolicy.normalize_request(
            pipeline_request(
                job_id="clean",
                track_id="track",
                user_id="user",
                job_type="magic_clean",
                audio_url="https://audio.test/a.mp3",
                storage=short_lived_storage,
            )
        )


def test_magic_clean_exact_replay_survives_credential_ttl_erosion(monkeypatch):
    short_lived_storage = {
        **STORAGE,
        "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
    }
    request = pipeline_request(
        job_id="clean",
        track_id="track",
        user_id="user",
        job_type="magic_clean",
        audio_url="https://audio.test/a.mp3",
        storage=short_lived_storage,
    )
    payload = SubmissionPolicy.normalize_request(request, enforce_magic_clean_storage_ttl=False)
    job = SimpleNamespace(
        id="clean",
        backend_id=BACKEND_ID,
        run_id="existing-run",
        track_id="track",
        job_type="magic_clean",
        status="completed",
        request_hash=SubmissionPolicy.request_fingerprint(payload),
        job_options=SubmissionPolicy._job_options(payload),
        input_url=payload["audio_url"],
        max_tags=payload["max_tags"],
        edited_transcript=None,
        custom_tags=None,
        storage_context_encrypted="encrypted-short-lived-context",
    )
    db = _FakeSubmissionDB(job)
    monkeypatch.setattr("hear.services.jobs.submission.DatabaseRuntime.SessionLocal", lambda: db)
    monkeypatch.setattr(
        "hear.services.jobs.submission.StorageContexts.encrypt_storage_context",
        lambda _storage: "encrypted-short-lived-context",
    )
    monkeypatch.setattr(
        "hear.services.jobs.submission.StorageContexts.decrypt_storage_context",
        lambda _token, **_kwargs: request.storage,
    )
    result = asyncio.run(JobSubmissionService(SimpleNamespace()).submit(request))
    assert result.replayed is True
    assert result.run_id == "existing-run"
    assert db.operations == ["execute", "query"]
    assert db.populated_existing is True
    assert db.locked_for_update is True
    assert db.rolled_back is False
    assert db.closed is True
    assert job.storage_context_encrypted == "encrypted-short-lived-context"


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
def test_terminal_magic_clean_replay_renews_cleanup_credentials(monkeypatch, status):
    original_request = pipeline_request(
        job_id="clean",
        track_id="track",
        user_id="user",
        job_type="magic_clean",
        audio_url="https://audio.test/a.mp3",
    )
    refreshed_request = pipeline_request(
        job_id="clean",
        track_id="track",
        user_id="user",
        job_type="magic_clean",
        audio_url="https://audio.test/a.mp3",
        storage={
            **STORAGE,
            "key_id": "rotated-key-id",
            "application_key": "rotated-application-key",
            "expires_at": (datetime.now(UTC) + timedelta(days=3)).isoformat(),
        },
    )
    refreshed_payload = SubmissionPolicy.normalize_request(refreshed_request)
    tombstone = {
        "b2_key": original_request.storage.folder_prefix + "enhanced/clean.mp3",
        "bucket_name": original_request.storage.bucket_name,
        "run_id": "existing-run",
    }
    job = _terminal_magic_clean_job(original_request, status=status, tombstone=tombstone)
    db = _FakeSubmissionDB(job)
    enqueue = AsyncMock()
    orchestrator = SimpleNamespace(enqueue=SimpleNamespace(remote=enqueue))
    monkeypatch.setattr("hear.services.jobs.submission.DatabaseRuntime.SessionLocal", lambda: db)
    monkeypatch.setattr(
        "hear.services.jobs.submission.StorageContexts.encrypt_storage_context",
        lambda _storage: "encrypted-refreshed-context",
    )
    monkeypatch.setattr(
        "hear.services.jobs.submission.StorageContexts.decrypt_storage_context",
        lambda _token, **_kwargs: original_request.storage,
    )
    result = asyncio.run(JobSubmissionService(orchestrator).submit(refreshed_request))
    assert result.replayed is True
    assert result.run_id == "existing-run"
    assert result.status == status
    assert job.storage_context_encrypted == "encrypted-refreshed-context"
    assert (
        job.job_options["storage_destination"]["expires_at"]
        == refreshed_payload["storage"]["expires_at"]
    )
    assert job.job_options["magic_clean_cleanup_tombstone"] == tombstone
    assert job.request_hash == SubmissionPolicy.request_fingerprint(refreshed_payload)
    assert job.error == "terminal-error"
    assert db.commits == 2
    assert db.rolled_back is False
    enqueue.assert_not_awaited()


@pytest.mark.parametrize(
    "tombstone",
    [
        None,
        {
            "b2_key": "users/user/jobs/job/enhanced/clean.mp3",
            "bucket_name": "backend-a-bucket",
            "run_id": "other-run",
        },
        {
            "b2_key": "users/user/jobs/job/enhanced/clean.mp3",
            "bucket_name": "other-bucket",
            "run_id": "existing-run",
        },
        {
            "b2_key": "users/other/jobs/job/enhanced/clean.mp3",
            "bucket_name": "backend-a-bucket",
            "run_id": "existing-run",
        },
        {
            "b2_key": "users/user/jobs/job/../other/clean.mp3",
            "bucket_name": "backend-a-bucket",
            "run_id": "existing-run",
        },
    ],
)
def test_terminal_magic_clean_rejects_renewal_without_owned_cleanup_tombstone(
    monkeypatch, tombstone
):
    original_request = pipeline_request(
        job_id="clean",
        track_id="track",
        user_id="user",
        job_type="magic_clean",
        audio_url="https://audio.test/a.mp3",
    )
    refreshed_request = pipeline_request(
        job_id="clean",
        track_id="track",
        user_id="user",
        job_type="magic_clean",
        audio_url="https://audio.test/a.mp3",
        storage={
            **STORAGE,
            "key_id": "rotated-key-id",
            "application_key": "rotated-application-key",
            "expires_at": (datetime.now(UTC) + timedelta(days=3)).isoformat(),
        },
    )
    job = _terminal_magic_clean_job(original_request, tombstone=tombstone)
    db = _FakeSubmissionDB(job)
    enqueue = AsyncMock()
    orchestrator = SimpleNamespace(enqueue=SimpleNamespace(remote=enqueue))
    monkeypatch.setattr("hear.services.jobs.submission.DatabaseRuntime.SessionLocal", lambda: db)
    monkeypatch.setattr(
        "hear.services.jobs.submission.StorageContexts.encrypt_storage_context",
        lambda _storage: "encrypted-refreshed-context",
    )
    monkeypatch.setattr(
        "hear.services.jobs.submission.StorageContexts.decrypt_storage_context",
        lambda _token, **_kwargs: original_request.storage,
    )
    with pytest.raises(ValueError, match="requires a valid cleanup tombstone"):
        asyncio.run(JobSubmissionService(orchestrator).submit(refreshed_request))
    assert job.storage_context_encrypted == "encrypted-original-context"
    assert db.commits == 1
    assert db.rolled_back is True
    enqueue.assert_not_awaited()


@pytest.mark.parametrize("rotate_key", [False, True])
def test_terminal_magic_clean_cleanup_credential_renewal_must_advance_expiry(
    monkeypatch, rotate_key
):
    original_storage = {
        **STORAGE,
        "expires_at": (datetime.now(UTC) + timedelta(days=4)).isoformat(),
    }
    original_request = pipeline_request(
        job_id="clean",
        track_id="track",
        user_id="user",
        job_type="magic_clean",
        audio_url="https://audio.test/a.mp3",
        storage=original_storage,
    )
    submitted_storage = {
        **original_storage,
        "expires_at": original_storage["expires_at"]
        if rotate_key
        else (datetime.now(UTC) + timedelta(days=3)).isoformat(),
    }
    if rotate_key:
        submitted_storage.update(key_id="rotated-key-id", application_key="rotated-application-key")
    submitted_request = pipeline_request(
        job_id="clean",
        track_id="track",
        user_id="user",
        job_type="magic_clean",
        audio_url="https://audio.test/a.mp3",
        storage=submitted_storage,
    )
    job = _terminal_magic_clean_job(
        original_request,
        tombstone={
            "b2_key": original_request.storage.folder_prefix + "enhanced/clean.mp3",
            "bucket_name": original_request.storage.bucket_name,
            "run_id": "existing-run",
        },
    )
    db = _FakeSubmissionDB(job)
    enqueue = AsyncMock()
    orchestrator = SimpleNamespace(enqueue=SimpleNamespace(remote=enqueue))
    monkeypatch.setattr("hear.services.jobs.submission.DatabaseRuntime.SessionLocal", lambda: db)
    monkeypatch.setattr(
        "hear.services.jobs.submission.StorageContexts.encrypt_storage_context",
        lambda _storage: "encrypted-submitted-context",
    )
    monkeypatch.setattr(
        "hear.services.jobs.submission.StorageContexts.decrypt_storage_context",
        lambda _token, **_kwargs: original_request.storage,
    )
    with pytest.raises(ValueError, match="must use a later expiration"):
        asyncio.run(JobSubmissionService(orchestrator).submit(submitted_request))
    assert job.storage_context_encrypted == "encrypted-original-context"
    assert db.rolled_back is True
    enqueue.assert_not_awaited()


def test_terminal_magic_clean_rejects_short_cleanup_credential_renewal(monkeypatch):
    original_storage = {
        **STORAGE,
        "expires_at": (datetime.now(UTC) + timedelta(minutes=30)).isoformat(),
    }
    original_request = pipeline_request(
        job_id="clean",
        track_id="track",
        user_id="user",
        job_type="magic_clean",
        audio_url="https://audio.test/a.mp3",
        storage=original_storage,
    )
    refreshed_request = pipeline_request(
        job_id="clean",
        track_id="track",
        user_id="user",
        job_type="magic_clean",
        audio_url="https://audio.test/a.mp3",
        storage={
            **STORAGE,
            "key_id": "rotated-key-id",
            "application_key": "rotated-application-key",
            "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        },
    )
    job = _terminal_magic_clean_job(
        original_request,
        tombstone={
            "b2_key": original_request.storage.folder_prefix + "enhanced/clean.mp3",
            "bucket_name": original_request.storage.bucket_name,
            "run_id": "existing-run",
        },
    )
    db = _FakeSubmissionDB(job)
    enqueue = AsyncMock()
    orchestrator = SimpleNamespace(enqueue=SimpleNamespace(remote=enqueue))
    monkeypatch.setattr("hear.services.jobs.submission.DatabaseRuntime.SessionLocal", lambda: db)
    monkeypatch.setattr(
        "hear.services.jobs.submission.StorageContexts.encrypt_storage_context",
        lambda _storage: "encrypted-refreshed-context",
    )
    monkeypatch.setattr(
        "hear.services.jobs.submission.StorageContexts.decrypt_storage_context",
        lambda _token, **_kwargs: original_request.storage,
    )
    with pytest.raises(StorageCredentialsExpiringError, match="must remain valid"):
        asyncio.run(JobSubmissionService(orchestrator).submit(refreshed_request))
    assert job.storage_context_encrypted == "encrypted-original-context"
    assert db.rolled_back is True
    enqueue.assert_not_awaited()


def test_terminal_magic_clean_cleanup_renewal_rejects_destination_change(monkeypatch):
    original_request = pipeline_request(
        job_id="clean",
        track_id="track",
        user_id="user",
        job_type="magic_clean",
        audio_url="https://audio.test/a.mp3",
    )
    changed_request = pipeline_request(
        job_id="clean",
        track_id="track",
        user_id="user",
        job_type="magic_clean",
        audio_url="https://audio.test/a.mp3",
        storage={
            **STORAGE,
            "folder_prefix": "users/user/jobs/other",
            "key_id": "rotated-key-id",
            "application_key": "rotated-application-key",
            "expires_at": (datetime.now(UTC) + timedelta(days=3)).isoformat(),
        },
    )
    job = _terminal_magic_clean_job(
        original_request,
        tombstone={
            "b2_key": original_request.storage.folder_prefix + "enhanced/clean.mp3",
            "bucket_name": original_request.storage.bucket_name,
            "run_id": "existing-run",
        },
    )
    db = _FakeSubmissionDB(job)
    enqueue = AsyncMock()
    orchestrator = SimpleNamespace(enqueue=SimpleNamespace(remote=enqueue))
    monkeypatch.setattr("hear.services.jobs.submission.DatabaseRuntime.SessionLocal", lambda: db)
    monkeypatch.setattr(
        "hear.services.jobs.submission.StorageContexts.encrypt_storage_context",
        lambda _storage: "encrypted-refreshed-context",
    )
    with pytest.raises(SubmissionConflictError, match="different payload"):
        asyncio.run(JobSubmissionService(orchestrator).submit(changed_request))
    assert job.storage_context_encrypted == "encrypted-original-context"
    assert db.rolled_back is True
    enqueue.assert_not_awaited()


def test_new_magic_clean_with_short_credentials_rolls_back_before_commit(monkeypatch):
    short_lived_storage = {
        **STORAGE,
        "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
    }
    request = pipeline_request(
        job_id="new-clean",
        track_id="track",
        user_id="user",
        job_type="magic_clean",
        audio_url="https://audio.test/a.mp3",
        storage=short_lived_storage,
    )
    db = _FakeSubmissionDB(None, inserted=True)
    monkeypatch.setattr("hear.services.jobs.submission.DatabaseRuntime.SessionLocal", lambda: db)
    monkeypatch.setattr(
        "hear.services.jobs.submission.StorageContexts.encrypt_storage_context",
        lambda _storage: "encrypted-short-lived-context",
    )
    with pytest.raises(StorageCredentialsExpiringError, match="must remain valid"):
        asyncio.run(JobSubmissionService(SimpleNamespace()).submit(request))
    assert db.operations == ["execute"]
    assert db.commits == 0
    assert db.rolled_back is True
    assert db.closed is True


def test_magic_clean_replay_can_refresh_nonterminal_credentials(monkeypatch):
    original_request = pipeline_request(
        job_id="clean",
        track_id="track",
        user_id="user",
        job_type="magic_clean",
        audio_url="https://audio.test/a.mp3",
    )
    original_payload = SubmissionPolicy.normalize_request(original_request)
    refreshed_storage = {
        **STORAGE,
        "key_id": "rotated-key-id",
        "application_key": "rotated-application-key",
        "expires_at": (datetime.now(UTC) + timedelta(days=3)).isoformat(),
    }
    refreshed_request = pipeline_request(
        job_id="clean",
        track_id="track",
        user_id="user",
        job_type="magic_clean",
        audio_url="https://audio.test/a.mp3",
        storage=refreshed_storage,
    )
    refreshed_payload = SubmissionPolicy.normalize_request(refreshed_request)
    job = SimpleNamespace(
        id="clean",
        backend_id=BACKEND_ID,
        run_id="existing-run",
        track_id="track",
        job_type="magic_clean",
        status="queued",
        request_hash=SubmissionPolicy.request_fingerprint(original_payload),
        job_options=SubmissionPolicy._job_options(original_payload),
        input_url=original_payload["audio_url"],
        max_tags=original_payload["max_tags"],
        edited_transcript=None,
        custom_tags=None,
        storage_context_encrypted="encrypted-original-context",
        error=None,
    )
    db = _FakeSubmissionDB(job)
    enqueue = AsyncMock()
    orchestrator = SimpleNamespace(enqueue=SimpleNamespace(remote=enqueue))
    monkeypatch.setattr("hear.services.jobs.submission.DatabaseRuntime.SessionLocal", lambda: db)
    monkeypatch.setattr(
        "hear.services.jobs.submission.StorageContexts.encrypt_storage_context",
        lambda _storage: "encrypted-refreshed-context",
    )
    monkeypatch.setattr(
        "hear.services.jobs.submission.StorageContexts.decrypt_storage_context",
        lambda _token, **_kwargs: original_request.storage,
    )
    result = asyncio.run(JobSubmissionService(orchestrator).submit(refreshed_request))
    assert result.replayed is True
    assert result.run_id == "existing-run"
    assert job.storage_context_encrypted == "encrypted-refreshed-context"
    assert (
        job.job_options["storage_destination"]["expires_at"]
        == refreshed_payload["storage"]["expires_at"]
    )
    assert job.request_hash == SubmissionPolicy.request_fingerprint(refreshed_payload)
    assert db.commits == 2
    enqueue.assert_awaited_once_with("clean", "existing-run")


def test_magic_clean_delayed_credential_retry_cannot_regress_refresh(monkeypatch):
    original_request = pipeline_request(
        job_id="clean",
        track_id="track",
        user_id="user",
        job_type="magic_clean",
        audio_url="https://audio.test/a.mp3",
    )
    original_payload = SubmissionPolicy.normalize_request(original_request)
    refreshed_storage = {
        **STORAGE,
        "key_id": "rotated-key-id",
        "application_key": "rotated-application-key",
        "expires_at": (datetime.now(UTC) + timedelta(days=3)).isoformat(),
    }
    refreshed_request = pipeline_request(
        job_id="clean",
        track_id="track",
        user_id="user",
        job_type="magic_clean",
        audio_url="https://audio.test/a.mp3",
        storage=refreshed_storage,
    )
    refreshed_payload = SubmissionPolicy.normalize_request(refreshed_request)
    job = SimpleNamespace(
        id="clean",
        backend_id=BACKEND_ID,
        run_id="existing-run",
        track_id="track",
        job_type="magic_clean",
        status="queued",
        request_hash=SubmissionPolicy.request_fingerprint(original_payload),
        job_options=SubmissionPolicy._job_options(original_payload),
        input_url=original_payload["audio_url"],
        max_tags=original_payload["max_tags"],
        edited_transcript=None,
        custom_tags=None,
        storage_context_encrypted="encrypted-original-context",
        error=None,
    )
    db = _FakeSubmissionDB(job)
    enqueue = AsyncMock()
    orchestrator = SimpleNamespace(enqueue=SimpleNamespace(remote=enqueue))
    monkeypatch.setattr("hear.services.jobs.submission.DatabaseRuntime.SessionLocal", lambda: db)
    monkeypatch.setattr(
        "hear.services.jobs.submission.StorageContexts.encrypt_storage_context",
        lambda storage: f"encrypted-{storage.key_id}",
    )
    monkeypatch.setattr(
        "hear.services.jobs.submission.StorageContexts.decrypt_storage_context",
        lambda token, **_kwargs: (
            original_request.storage
            if token == "encrypted-original-context"
            else refreshed_request.storage
        ),
    )
    service = JobSubmissionService(orchestrator)
    asyncio.run(service.submit(refreshed_request))
    with pytest.raises(ValueError, match="rotation must use a later expiration"):
        asyncio.run(service.submit(original_request))
    assert job.storage_context_encrypted == "encrypted-rotated-key-id"
    assert (
        job.job_options["storage_destination"]["expires_at"]
        == refreshed_payload["storage"]["expires_at"]
    )
    assert job.request_hash == SubmissionPolicy.request_fingerprint(refreshed_payload)
    enqueue.assert_awaited_once_with("clean", "existing-run")


def test_magic_clean_rejects_unsafe_nonterminal_credential_refresh(monkeypatch):
    original_storage = {
        **STORAGE,
        "expires_at": (datetime.now(UTC) + timedelta(minutes=30)).isoformat(),
    }
    original_request = pipeline_request(
        job_id="clean",
        track_id="track",
        user_id="user",
        job_type="magic_clean",
        audio_url="https://audio.test/a.mp3",
        storage=original_storage,
    )
    original_payload = SubmissionPolicy.normalize_request(
        original_request, enforce_magic_clean_storage_ttl=False
    )
    short_lived_storage = {
        **STORAGE,
        "key_id": "rotated-key-id",
        "application_key": "rotated-application-key",
        "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
    }
    short_lived_request = pipeline_request(
        job_id="clean",
        track_id="track",
        user_id="user",
        job_type="magic_clean",
        audio_url="https://audio.test/a.mp3",
        storage=short_lived_storage,
    )
    job = SimpleNamespace(
        id="clean",
        backend_id=BACKEND_ID,
        run_id="existing-run",
        track_id="track",
        job_type="magic_clean",
        status="queued",
        request_hash=SubmissionPolicy.request_fingerprint(original_payload),
        job_options=SubmissionPolicy._job_options(original_payload),
        input_url=original_payload["audio_url"],
        max_tags=original_payload["max_tags"],
        edited_transcript=None,
        custom_tags=None,
        storage_context_encrypted="encrypted-original-context",
        error="storage_credentials_expiring",
    )
    db = _FakeSubmissionDB(job)
    enqueue = AsyncMock()
    orchestrator = SimpleNamespace(enqueue=SimpleNamespace(remote=enqueue))
    monkeypatch.setattr("hear.services.jobs.submission.DatabaseRuntime.SessionLocal", lambda: db)
    monkeypatch.setattr(
        "hear.services.jobs.submission.StorageContexts.encrypt_storage_context",
        lambda _storage: "encrypted-short-lived-context",
    )
    monkeypatch.setattr(
        "hear.services.jobs.submission.StorageContexts.decrypt_storage_context",
        lambda _token, **_kwargs: original_request.storage,
    )
    with pytest.raises(StorageCredentialsExpiringError, match="must remain valid"):
        asyncio.run(JobSubmissionService(orchestrator).submit(short_lived_request))
    assert job.storage_context_encrypted == "encrypted-original-context"
    assert db.commits == 1
    assert db.rolled_back is True
    enqueue.assert_not_awaited()


def test_magic_clean_unsafe_exact_queued_replay_stays_parked(monkeypatch):
    short_lived_storage = {
        **STORAGE,
        "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
    }
    request = pipeline_request(
        job_id="clean",
        track_id="track",
        user_id="user",
        job_type="magic_clean",
        audio_url="https://audio.test/a.mp3",
        storage=short_lived_storage,
    )
    payload = SubmissionPolicy.normalize_request(request, enforce_magic_clean_storage_ttl=False)
    job = SimpleNamespace(
        id="clean",
        backend_id=BACKEND_ID,
        run_id="existing-run",
        track_id="track",
        job_type="magic_clean",
        status="queued",
        request_hash=SubmissionPolicy.request_fingerprint(payload),
        job_options=SubmissionPolicy._job_options(payload),
        input_url=payload["audio_url"],
        max_tags=payload["max_tags"],
        edited_transcript=None,
        custom_tags=None,
        storage_context_encrypted="encrypted-short-lived-context",
        error="storage_credentials_expiring",
    )
    db = _FakeSubmissionDB(job)
    enqueue = AsyncMock()
    orchestrator = SimpleNamespace(enqueue=SimpleNamespace(remote=enqueue))
    monkeypatch.setattr("hear.services.jobs.submission.DatabaseRuntime.SessionLocal", lambda: db)
    monkeypatch.setattr(
        "hear.services.jobs.submission.StorageContexts.encrypt_storage_context",
        lambda _storage: "encrypted-short-lived-context",
    )
    monkeypatch.setattr(
        "hear.services.jobs.submission.StorageContexts.decrypt_storage_context",
        lambda _token, **_kwargs: request.storage,
    )
    result = asyncio.run(JobSubmissionService(orchestrator).submit(request))
    assert result.replayed is True
    assert result.run_id == "existing-run"
    assert result.status == "queued"
    assert job.status == "queued"
    assert job.error == "storage_credentials_expiring"
    assert job.storage_context_encrypted == "encrypted-short-lived-context"
    assert db.commits == 2
    assert db.rolled_back is False
    enqueue.assert_not_awaited()


def test_magic_clean_key_rotation_requires_later_expiration(monkeypatch):
    original_request = pipeline_request(
        job_id="clean",
        track_id="track",
        user_id="user",
        job_type="magic_clean",
        audio_url="https://audio.test/a.mp3",
    )
    payload = SubmissionPolicy.normalize_request(original_request)
    rotated_request = pipeline_request(
        job_id="clean",
        track_id="track",
        user_id="user",
        job_type="magic_clean",
        audio_url="https://audio.test/a.mp3",
        storage={
            **STORAGE,
            "key_id": "rotated-key-id",
            "application_key": "rotated-application-key",
        },
    )
    job = SimpleNamespace(
        id="clean",
        backend_id=BACKEND_ID,
        run_id="existing-run",
        track_id="track",
        job_type="magic_clean",
        status="queued",
        request_hash=SubmissionPolicy.request_fingerprint(payload),
        job_options=SubmissionPolicy._job_options(payload),
        input_url=payload["audio_url"],
        max_tags=payload["max_tags"],
        edited_transcript=None,
        custom_tags=None,
        storage_context_encrypted="encrypted-original-context",
        error=None,
    )
    db = _FakeSubmissionDB(job)
    enqueue = AsyncMock()
    orchestrator = SimpleNamespace(enqueue=SimpleNamespace(remote=enqueue))
    monkeypatch.setattr("hear.services.jobs.submission.DatabaseRuntime.SessionLocal", lambda: db)
    monkeypatch.setattr(
        "hear.services.jobs.submission.StorageContexts.encrypt_storage_context",
        lambda _storage: "encrypted-rotated-context",
    )
    monkeypatch.setattr(
        "hear.services.jobs.submission.StorageContexts.decrypt_storage_context",
        lambda _token, **_kwargs: original_request.storage,
    )
    rotated_payload = SubmissionPolicy.normalize_request(rotated_request)
    assert SubmissionPolicy.request_fingerprint(
        rotated_payload
    ) == SubmissionPolicy.request_fingerprint(payload)
    with pytest.raises(ValueError, match="rotation must use a later expiration"):
        asyncio.run(JobSubmissionService(orchestrator).submit(rotated_request))
    assert job.storage_context_encrypted == "encrypted-original-context"
    enqueue.assert_not_awaited()


def test_magic_clean_running_job_rejects_credential_refresh(monkeypatch):
    original_request = pipeline_request(
        job_id="clean",
        track_id="track",
        user_id="user",
        job_type="magic_clean",
        audio_url="https://audio.test/a.mp3",
    )
    payload = SubmissionPolicy.normalize_request(original_request)
    refreshed_request = pipeline_request(
        job_id="clean",
        track_id="track",
        user_id="user",
        job_type="magic_clean",
        audio_url="https://audio.test/a.mp3",
        storage={
            **STORAGE,
            "key_id": "rotated-key-id",
            "application_key": "rotated-application-key",
            "expires_at": (datetime.now(UTC) + timedelta(days=3)).isoformat(),
        },
    )
    job = SimpleNamespace(
        id="clean",
        backend_id=BACKEND_ID,
        run_id="existing-run",
        track_id="track",
        job_type="magic_clean",
        status="running",
        request_hash=SubmissionPolicy.request_fingerprint(payload),
        job_options=SubmissionPolicy._job_options(payload),
        input_url=payload["audio_url"],
        max_tags=payload["max_tags"],
        edited_transcript=None,
        custom_tags=None,
        storage_context_encrypted="encrypted-original-context",
        error=None,
    )
    db = _FakeSubmissionDB(job)
    monkeypatch.setattr("hear.services.jobs.submission.DatabaseRuntime.SessionLocal", lambda: db)
    monkeypatch.setattr(
        "hear.services.jobs.submission.StorageContexts.encrypt_storage_context",
        lambda _storage: "encrypted-refreshed-context",
    )
    monkeypatch.setattr(
        "hear.services.jobs.submission.StorageContexts.decrypt_storage_context",
        lambda _token, **_kwargs: original_request.storage,
    )
    with pytest.raises(ValueError, match="cannot be refreshed after the job has started"):
        asyncio.run(JobSubmissionService(SimpleNamespace()).submit(refreshed_request))
    assert job.status == "running"
    assert job.storage_context_encrypted == "encrypted-original-context"


def test_magic_clean_records_omitted_controls_without_changing_idempotency():
    omitted = SubmissionPolicy.normalize_request(
        pipeline_request(
            job_id="job",
            track_id="track",
            user_id="user",
            job_type="magic_clean",
            audio_url="https://audio.test/a.mp3",
        )
    )
    explicit_default = SubmissionPolicy.normalize_request(
        pipeline_request(
            job_id="job",
            track_id="track",
            user_id="user",
            job_type="magic_clean",
            audio_url="https://audio.test/a.mp3",
            speech=100,
            music=10,
            background=10,
        )
    )
    assert omitted["magic_clean_controls_omitted"] is True
    assert explicit_default["magic_clean_controls_omitted"] is False
    assert SubmissionPolicy.request_fingerprint(omitted) == SubmissionPolicy.request_fingerprint(
        explicit_default
    )


def test_magic_clean_silence_option_is_normalized_and_fingerprinted():
    keep = SubmissionPolicy.normalize_request(
        pipeline_request(
            job_id="job",
            track_id="track",
            user_id="user",
            job_type="magic_clean",
            audio_url="https://audio.test/a.mp3",
        )
    )
    cut = SubmissionPolicy.normalize_request(
        pipeline_request(
            job_id="job",
            track_id="track",
            user_id="user",
            job_type="magic_clean",
            audio_url="https://audio.test/a.mp3",
            cut_silence=True,
        )
    )
    assert keep["cut_silence"] is False
    assert cut["cut_silence"] is True
    assert SubmissionPolicy.request_fingerprint(cut) != SubmissionPolicy.request_fingerprint(keep)


@pytest.mark.parametrize("field", ["speech", "music", "background"])
def test_magic_clean_stem_levels_must_be_percentages(field):
    with pytest.raises(ValueError):
        pipeline_request(job_id="job", track_id="track", user_id="user", **{field: 101})


def test_magic_clean_stem_levels_must_be_supplied_together():
    with pytest.raises(ValueError, match="supplied together"):
        pipeline_request(job_id="job", track_id="track", user_id="user", speech=50)


@pytest.mark.parametrize("job_type", ["rebuild", "edit_transcript"])
def test_transcript_jobs_require_edited_transcript(job_type):
    with pytest.raises(ValueError, match="edited_transcript"):
        SubmissionPolicy.normalize_request(
            pipeline_request(job_id="job", track_id="track", user_id="user", job_type=job_type)
        )


def test_reconstruct_requires_valid_changes():
    with pytest.raises(ValueError, match="changes"):
        SubmissionPolicy.normalize_request(
            pipeline_request(job_id="job", track_id="track", user_id="user", job_type="reconstruct")
        )
    with pytest.raises(ValueError, match="end after"):
        SubmissionPolicy.normalize_request(
            pipeline_request(
                job_id="job",
                track_id="track",
                user_id="user",
                job_type="reconstruct",
                audio_url="https://audio.test/a.mp3",
                changes=[SegmentChange(segment_start=2, segment_end=1, new_text="replacement")],
            )
        )


def test_legacy_job_without_storage_cannot_match_current_request():
    job = AiJob(
        id="job",
        run_id="run",
        track_id="track",
        job_type="pipeline",
        max_tags=8,
        status="completed",
        input_url="https://audio.test/a.mp3",
        job_options={
            "grouped": False,
            "kind": "track",
            "track_count": 1,
            "user_id": "user",
        },
    )
    request = SubmissionPolicy.normalize_request(
        pipeline_request(
            job_id="job", track_id="track", user_id="user", audio_url="https://audio.test/a.mp3"
        )
    )
    assert SubmissionPolicy.request_fingerprint(
        SubmissionPolicy._legacy_payload(job)
    ) != SubmissionPolicy.request_fingerprint(request)


def test_user_id_is_normalized_and_cannot_be_blank():
    request = SubmissionPolicy.normalize_request(
        pipeline_request(
            job_id="job",
            track_id="track",
            user_id="  user-1  ",
            audio_url="https://audio.test/a.mp3",
        )
    )
    assert request["user_id"] == "user-1"
    with pytest.raises(ValueError, match="user_id is required"):
        SubmissionPolicy.normalize_request(
            pipeline_request(
                job_id="job", track_id="track", user_id="   ", audio_url="https://audio.test/a.mp3"
            )
        )
