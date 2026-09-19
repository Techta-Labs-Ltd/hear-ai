from __future__ import annotations

import asyncio
import threading
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hear.config import settings
from hear.core.storage import (
    StorageCredentialsExpiredError,
    StorageCredentialsExpiringError,
)
from hear.models.database import AiTrackJob
from hear.orchestrator import (
    NON_RETRYABLE_JOB_ERRORS,
    RECOVERY_RETRY_LIMIT_ERROR,
    Orchestrator,
)
from hear.services.jobs.scheduler import FairJobScheduler, PendingJob
from hear.services.magic_clean.lineage import (
    MAGIC_CLEAN_DELIVERED_FILE_SHA256_KEY,
    MAGIC_CLEAN_DELIVERED_PCM_SHA256_KEY,
    MAGIC_CLEAN_ENGINE_REVISION_KEY,
    MAGIC_CLEAN_ROOT_URL_KEY,
    MAGIC_CLEAN_SOURCE_FILE_SHA256_KEY,
    MAGIC_CLEAN_SOURCE_PCM_SHA256_KEY,
    MagicCleanLineageError,
)
from hear.services.magic_clean.processing.validation import AudioValidationError

FILE_HASH = "a" * 64
PCM_HASH = "b" * 64
DELIVERED_FILE_HASH = "c" * 64
DELIVERED_PCM_HASH = "d" * 64
LEVELS = {"speech": 100, "music": 10, "background": 10, "cut_silence": False}


def test_delivery_integrity_failures_are_not_retried() -> None:
    assert isinstance(AudioValidationError("invalid delivery"), NON_RETRYABLE_JOB_ERRORS)


def test_magic_clean_runtime_requires_full_storage_credential_reserve(
    monkeypatch,
) -> None:
    cls = Orchestrator.func_or_class
    monkeypatch.setattr(
        settings,
        "MAGIC_CLEAN_STORAGE_CREDENTIAL_MIN_TTL_SECONDS",
        3600,
    )

    cls._require_magic_clean_storage_lifetime(
        SimpleNamespace(
            context=SimpleNamespace(
                expires_at=datetime.now(UTC) + timedelta(hours=2)
            )
        )
    )
    with pytest.raises(StorageCredentialsExpiringError):
        cls._require_magic_clean_storage_lifetime(
            SimpleNamespace(
                context=SimpleNamespace(
                    expires_at=datetime.now(UTC) + timedelta(minutes=30)
                )
            )
        )


def test_scheduler_parks_magic_clean_with_insufficient_credential_reserve(
    monkeypatch,
) -> None:
    cls = Orchestrator.func_or_class
    orchestrator = cls.__new__(cls)
    events = []
    orchestrator._push_event = lambda _job_id, event: events.append(event)
    job = SimpleNamespace(
        id="job",
        run_id="run",
        status="queued",
        job_type="magic_clean",
        track_id="track",
        attempts=2,
        current_stage=None,
        error=None,
        storage_context_encrypted="encrypted-context",
        job_options={"user_id": "user"},
    )
    track_job = SimpleNamespace(
        status="running",
        attempts=1,
        current_stage="downloading",
        error=None,
        updated_at=None,
    )

    class Query:
        def __init__(self, model):
            self.model = model

        def filter(self, *_args):
            return self

        def with_for_update(self):
            return self

        def first(self):
            return job if self.model is not AiTrackJob else track_job

    class Session:
        def __init__(self):
            self.commits = 0
            self.closed = False

        @staticmethod
        def query(model):
            return Query(model)

        def commit(self):
            self.commits += 1

        def close(self):
            self.closed = True

    session = Session()
    monkeypatch.setattr("hear.orchestrator.SessionLocal", lambda: session)
    monkeypatch.setattr(
        "hear.orchestrator.decrypt_storage_context",
        lambda _token, **_kwargs: SimpleNamespace(
            expires_at=datetime.now(UTC) + timedelta(minutes=30)
        ),
    )
    monkeypatch.setattr(
        settings,
        "MAGIC_CLEAN_STORAGE_CREDENTIAL_MIN_TTL_SECONDS",
        3600,
    )

    assert orchestrator._pending_job("job", "run") is None
    assert job.status == "queued"
    assert job.current_stage is None
    assert job.attempts == 2
    assert job.error == "storage_credentials_expiring"
    assert track_job.status == "queued"
    assert track_job.current_stage is None
    assert track_job.attempts == 1
    assert track_job.error == "storage_credentials_expiring"
    assert session.commits == 1
    assert session.closed is True
    assert events[0]["status"] == "queued"
    assert events[0]["error"] == "storage_credentials_expiring"
    assert events[0]["result"]["report"]["stage"] == "downloading"


async def _exercise_claim_rechecks_stale_magic_clean_storage(monkeypatch):
    cls = Orchestrator.func_or_class
    orchestrator = cls.__new__(cls)
    orchestrator._active_count = 0
    orchestrator._job_start_times = {}
    events = []
    orchestrator._push_event = lambda _job_id, event: events.append(event)
    job = SimpleNamespace(
        id="job",
        run_id="run",
        status="queued",
        job_type="magic_clean",
        track_id="track",
        attempts=0,
        current_stage=None,
        error=None,
        storage_context_encrypted="encrypted-context",
    )
    track_job = SimpleNamespace(
        status="queued",
        attempts=0,
        current_stage=None,
        error=None,
        updated_at=None,
    )

    class Query:
        def __init__(self, model):
            self.model = model

        def filter(self, *_args):
            return self

        def with_for_update(self):
            return self

        def first(self):
            return track_job if self.model is AiTrackJob else job

    class Session:
        def __init__(self):
            self.commits = 0
            self.closed = False

        @staticmethod
        def query(model):
            return Query(model)

        def commit(self):
            self.commits += 1

        def close(self):
            self.closed = True

    session = Session()
    monkeypatch.setattr("hear.orchestrator.SessionLocal", lambda: session)
    monkeypatch.setattr(
        "hear.orchestrator.decrypt_storage_context",
        lambda _token, **_kwargs: SimpleNamespace(
            expires_at=datetime.now(UTC) + timedelta(minutes=30)
        ),
    )
    monkeypatch.setattr(
        settings,
        "MAGIC_CLEAN_STORAGE_CREDENTIAL_MIN_TTL_SECONDS",
        3600,
    )
    monkeypatch.setattr("hear.orchestrator.cleanup_job_temp", lambda *_args: None)

    await orchestrator._process("job", "run")

    assert job.status == "queued"
    assert job.attempts == 0
    assert job.error == "storage_credentials_expiring"
    assert track_job.status == "queued"
    assert track_job.attempts == 0
    assert track_job.error == "storage_credentials_expiring"
    assert orchestrator._active_count == 0
    assert session.commits == 1
    assert session.closed is True
    assert events[0]["error"] == "storage_credentials_expiring"


def test_claim_rechecks_magic_clean_ttl_before_running_or_incrementing(
    monkeypatch,
) -> None:
    asyncio.run(_exercise_claim_rechecks_stale_magic_clean_storage(monkeypatch))


async def _exercise_stale_fair_queue_entry_skips_processing_slot():
    cls = Orchestrator.func_or_class
    orchestrator = cls.__new__(cls)
    pending = PendingJob("job", "run", "user", "magic_clean")
    orchestrator._pending_job = lambda *_args: None
    orchestrator.process = AsyncMock()
    orchestrator._finish_scheduled_run = lambda *_args: None

    await orchestrator._run_scheduled(pending)

    orchestrator.process.assert_not_awaited()


def test_stale_fair_queue_entry_is_rechecked_before_processing_slot() -> None:
    asyncio.run(_exercise_stale_fair_queue_entry_skips_processing_slot())


async def _exercise_subscribe_replays_credential_park(monkeypatch):
    cls = Orchestrator.func_or_class
    orchestrator = cls.__new__(cls)
    orchestrator._event_queues = {}
    job = SimpleNamespace(
        id="job",
        run_id="run",
        backend_id="backend",
        track_id="track",
        status="queued",
        job_type="magic_clean",
        current_stage=None,
        error="storage_credentials_expiring",
    )

    class Query:
        @staticmethod
        def filter(*_args):
            return Query()

        @staticmethod
        def first():
            return job

    class Session:
        def __init__(self):
            self.closed = False

        @staticmethod
        def query(*_args):
            return Query()

        def close(self):
            self.closed = True

    session = Session()
    monkeypatch.setattr("hear.orchestrator.SessionLocal", lambda: session)

    stream = orchestrator.subscribe("job")
    event = await anext(stream)
    await stream.aclose()

    assert event["event"] == "job_queued"
    assert event["status"] == "queued"
    assert event["error"] == "storage_credentials_expiring"
    assert session.closed is True


def test_subscribe_replays_queued_credential_park(monkeypatch) -> None:
    asyncio.run(_exercise_subscribe_replays_credential_park(monkeypatch))


async def _exercise_storage_refresh_parking(monkeypatch):
    cls = Orchestrator.func_or_class
    orchestrator = cls.__new__(cls)
    job = SimpleNamespace(
        id="job",
        run_id="run",
        status="running",
        job_type="magic_clean",
        track_id="track",
        attempts=2,
        current_stage="separating",
        error=None,
    )
    track_job = SimpleNamespace(
        id="track-job",
        status="running",
        current_stage="separating",
        error=None,
        updated_at=None,
    )

    class Query:
        def __init__(self, model):
            self.model = model

        def filter(self, *_args):
            return self

        def first(self):
            return job if self.model is not AiTrackJob else track_job

        def update(self, values, **_kwargs):
            target = job if self.model is not AiTrackJob else track_job
            for attribute, value in values.items():
                setattr(target, attribute.key, value)
            return 1

    class Session:
        committed = False
        rolled_back = False
        closed = False

        @staticmethod
        def query(model):
            return Query(model)

        def rollback(self):
            self.rolled_back = True

        def close(self):
            self.closed = True

    session = Session()

    async def commit(candidate):
        candidate.committed = True

    monkeypatch.setattr("hear.orchestrator.SessionLocal", lambda: session)
    monkeypatch.setattr("hear.orchestrator.commit_with_retry", commit)

    event = await orchestrator._park_magic_clean_for_storage_refresh(
        "job",
        "run",
    )

    assert event["error"] == "storage_credentials_expiring"
    assert event["result"]["report"]["stage"] == "separating"
    assert job.status == "queued"
    assert job.current_stage is None
    assert job.attempts == 1
    assert job.error == "storage_credentials_expiring"
    assert track_job.status == "queued"
    assert track_job.current_stage is None
    assert track_job.error == "storage_credentials_expiring"
    assert session.committed is True
    assert session.rolled_back is False
    assert session.closed is True


def test_runtime_ttl_guard_parks_job_without_consuming_attempt(monkeypatch):
    asyncio.run(_exercise_storage_refresh_parking(monkeypatch))


async def _exercise_process_parks_expired_storage(monkeypatch):
    cls = Orchestrator.func_or_class
    orchestrator = cls.__new__(cls)
    orchestrator._active_count = 0
    orchestrator._job_start_times = {}
    events = []
    orchestrator._push_event = lambda _job_id, event: events.append(event)
    parked_event = {
        "event": "job_queued",
        "job_id": "job",
        "status": "queued",
    }
    orchestrator._park_magic_clean_for_storage_refresh = AsyncMock(
        return_value=parked_event
    )
    job = SimpleNamespace(
        id="job",
        run_id="run",
        status="queued",
        job_type="magic_clean",
        attempts=0,
    )

    class Query:
        def filter(self, *_args):
            return self

        def with_for_update(self):
            return self

        @staticmethod
        def update(_values, **_kwargs):
            job.status = "running"
            job.attempts = 1
            return 1

        @staticmethod
        def first():
            return job

    class Session:
        rolled_back = False
        closed = False

        @staticmethod
        def query(*_args):
            return Query()

        def rollback(self):
            self.rolled_back = True

        def close(self):
            self.closed = True

    session = Session()

    async def commit(_session):
        return None

    monkeypatch.setattr("hear.orchestrator.SessionLocal", lambda: session)
    monkeypatch.setattr("hear.orchestrator.commit_with_retry", commit)
    monkeypatch.setattr(
        orchestrator,
        "_queued_magic_clean_storage_is_ready",
        lambda *_args: True,
    )
    monkeypatch.setattr(
        "hear.orchestrator.storage_for_job",
        lambda _job: (_ for _ in ()).throw(
            StorageCredentialsExpiringError("storage_credentials_expiring")
        ),
    )
    monkeypatch.setattr("hear.orchestrator.cleanup_job_temp", lambda *_args: None)

    await orchestrator._process("job", "run")

    orchestrator._park_magic_clean_for_storage_refresh.assert_awaited_once_with(
        "job",
        "run",
    )
    assert session.rolled_back is True
    assert session.closed is True
    assert orchestrator._active_count == 0
    assert events == [parked_event]


def test_process_parks_ttl_failure_instead_of_marking_job_failed(monkeypatch):
    asyncio.run(_exercise_process_parks_expired_storage(monkeypatch))


async def _exercise_non_magic_expired_storage_uses_normal_failure(monkeypatch):
    cls = Orchestrator.func_or_class
    orchestrator = cls.__new__(cls)
    orchestrator._active_count = 0
    orchestrator._job_start_times = {}
    events = []
    orchestrator._push_event = lambda _job_id, event: events.append(event)
    orchestrator._park_magic_clean_for_storage_refresh = AsyncMock(return_value=None)
    job = SimpleNamespace(
        id="job",
        run_id="run",
        backend_id="backend",
        track_id="track",
        status="queued",
        job_type="transcription",
        attempts=0,
        current_stage="downloading",
        error=None,
        completed_at=None,
    )
    track_job = SimpleNamespace(
        id="track-job",
        job_id="job",
        run_id="run",
        track_id="track",
        status="running",
        current_stage="downloading",
        error=None,
        completed_at=None,
        updated_at=None,
    )

    class Query:
        def __init__(self, model):
            self.model = model

        def filter(self, *_args):
            return self

        def with_for_update(self):
            return self

        def first(self):
            return track_job if self.model is AiTrackJob else job

        def update(self, values, **_kwargs):
            target = track_job if self.model is AiTrackJob else job
            for attribute, value in values.items():
                setattr(target, attribute.key, value)
            return 1

    class Session:
        def __init__(self):
            self.commits = 0
            self.rollbacks = 0
            self.closed = False

        @staticmethod
        def query(model):
            return Query(model)

        def rollback(self):
            self.rollbacks += 1

        def close(self):
            self.closed = True

    process_db = Session()
    failure_db = Session()
    sessions = iter((process_db, failure_db))

    async def commit(candidate):
        candidate.commits += 1

    monkeypatch.setattr("hear.orchestrator.SessionLocal", lambda: next(sessions))
    monkeypatch.setattr("hear.orchestrator.commit_with_retry", commit)
    monkeypatch.setattr(
        "hear.orchestrator.storage_for_job",
        lambda _job: (_ for _ in ()).throw(
            StorageCredentialsExpiredError("storage_credentials_expired")
        ),
    )
    monkeypatch.setattr("hear.orchestrator.cleanup_job_temp", lambda *_args: None)
    monkeypatch.setattr("hear.orchestrator.sentry_sdk.capture_exception", lambda *_args: None)

    await orchestrator._process("job", "run")

    orchestrator._park_magic_clean_for_storage_refresh.assert_not_awaited()
    assert job.status == "failed"
    assert job.attempts == 1
    assert job.current_stage is None
    assert job.error == "storage_credentials_expired"
    assert job.completed_at is not None
    assert track_job.status == "failed"
    assert track_job.current_stage is None
    assert track_job.error == "storage_credentials_expired"
    assert track_job.completed_at == job.completed_at
    assert process_db.closed is True
    assert failure_db.closed is True
    assert orchestrator._active_count == 0
    assert events[0]["event"] == "job_failed"
    assert events[0]["error"] == "storage_credentials_expired"


def test_non_magic_expired_storage_is_terminalized_by_normal_failure(
    monkeypatch,
) -> None:
    asyncio.run(_exercise_non_magic_expired_storage_uses_normal_failure(monkeypatch))


async def _exercise_recovery_terminalizes_exhausted_job(monkeypatch, status: str):
    cls = Orchestrator.func_or_class
    orchestrator = cls.__new__(cls)
    orchestrator._recovery_started = False
    scheduled = []
    orchestrator._schedule_job = lambda job_id, run_id: scheduled.append(
        (job_id, run_id)
    )
    old_not_before = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    tombstone = {
        "b2_key": "prefix/enhanced/job.mp3",
        "run_id": "run",
        "reason": "upload_pending",
        "last_error_type": "ProvisionalArtifact",
        "not_before": old_not_before,
    }
    job = SimpleNamespace(
        id="job",
        run_id="run",
        status=status,
        job_type="magic_clean",
        attempts=3,
        current_stage="finalizing",
        error=None,
        completed_at=None,
        job_options={"magic_clean_cleanup_tombstone": tombstone},
    )
    track_job = SimpleNamespace(
        job_id="job",
        run_id="run",
        status=status,
        current_stage="finalizing",
        error=None,
        completed_at=None,
        updated_at=None,
    )

    class Query:
        def __init__(self, entities):
            self.entities = entities

        def filter(self, *_args):
            return self

        def order_by(self, *_args):
            return self

        def with_for_update(self):
            session.locked_selects += 1
            return self

        def all(self):
            if len(self.entities) == 5:
                return [
                    (
                        job.id,
                        job.run_id,
                        job.status,
                        job.job_type,
                        job.job_options,
                    )
                ]
            if len(self.entities) == 3:
                return []
            raise AssertionError("unexpected recovery query")

        def update(self, values, **_kwargs):
            target = track_job if self.entities[0] is AiTrackJob else job
            for attribute, value in values.items():
                setattr(target, attribute.key, value)
            return 1

    class Session:
        def __init__(self):
            self.commits = 0
            self.closed = False
            self.locked_selects = 0

        @staticmethod
        def query(*entities):
            return Query(entities)

        def close(self):
            self.closed = True

    session = Session()

    async def commit(candidate):
        candidate.commits += 1

    monkeypatch.setattr(settings, "JOB_MAX_RETRIES", 3)
    monkeypatch.setattr(settings, "MAGIC_CLEAN_CLEANUP_GRACE_SECONDS", 120)
    monkeypatch.setattr("hear.orchestrator.SessionLocal", lambda: session)
    monkeypatch.setattr("hear.orchestrator.commit_with_retry", commit)

    await orchestrator.recover_jobs()

    assert job.status == "failed"
    assert job.current_stage is None
    assert job.error == RECOVERY_RETRY_LIMIT_ERROR
    assert job.completed_at is not None
    persisted_tombstone = job.job_options["magic_clean_cleanup_tombstone"]
    assert persisted_tombstone["b2_key"] == tombstone["b2_key"]
    assert persisted_tombstone["run_id"] == tombstone["run_id"]
    assert persisted_tombstone["reason"] == tombstone["reason"]
    assert persisted_tombstone["last_error_type"] == tombstone["last_error_type"]
    if status == "running":
        assert persisted_tombstone["not_before"] != old_not_before
        refreshed_not_before = datetime.fromisoformat(persisted_tombstone["not_before"])
        assert refreshed_not_before > datetime.now(UTC) + timedelta(seconds=115)
    else:
        assert persisted_tombstone["not_before"] == old_not_before
    assert track_job.status == "failed"
    assert track_job.current_stage is None
    assert track_job.error == RECOVERY_RETRY_LIMIT_ERROR
    assert track_job.completed_at == job.completed_at
    assert track_job.updated_at == job.completed_at
    assert scheduled == []
    assert session.commits == 1
    assert session.locked_selects == 2
    assert session.closed is True


@pytest.mark.parametrize("status", ["queued", "running"])
def test_recovery_terminalizes_retry_exhausted_jobs_for_cleanup_reconciliation(
    monkeypatch,
    status,
) -> None:
    asyncio.run(_exercise_recovery_terminalizes_exhausted_job(monkeypatch, status))


async def _exercise_active_scheduled_run_cancellation():
    cls = Orchestrator.func_or_class
    orchestrator = cls.__new__(cls)
    removed: list[str] = []
    orchestrator._fair_scheduler = SimpleNamespace(
        remove=lambda job_id: removed.append(job_id)
    )
    orchestrator._scheduled_runs = {("job", "run")}
    orchestrator._run_tasks = {}
    orchestrator._dispatch_event = asyncio.Event()
    started = asyncio.Event()

    async def active_run():
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(active_run())
    orchestrator._run_tasks[("job", "run")] = task
    await started.wait()

    orchestrator._cancel_scheduled_run("job", "run")

    with pytest.raises(asyncio.CancelledError):
        await task
    assert removed == ["job"]
    assert ("job", "run") not in orchestrator._scheduled_runs
    assert orchestrator._dispatch_event.is_set()


def test_active_scheduled_run_is_signalled_on_cancel() -> None:
    asyncio.run(_exercise_active_scheduled_run_cancellation())


async def _exercise_cancel_before_run_coroutine_starts():
    cls = Orchestrator.func_or_class
    orchestrator = cls.__new__(cls)
    scheduler = FairJobScheduler(
        max_active=1,
        max_active_per_user=1,
        type_limits={"magic_clean": 1},
    )
    pending = PendingJob("job", "run", "user", "magic_clean")
    assert scheduler.enqueue(pending)
    assert scheduler.pop_next() == pending
    orchestrator._fair_scheduler = scheduler
    orchestrator._scheduled_runs = {pending.key}
    orchestrator._run_tasks = {}
    orchestrator._dispatch_event = asyncio.Event()
    orchestrator.process = AsyncMock(
        side_effect=AssertionError("cancelled run must not start")
    )

    task = orchestrator._start_scheduled_run(pending)
    orchestrator._cancel_scheduled_run(pending.job_id, pending.run_id)

    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    orchestrator.process.assert_not_awaited()
    assert scheduler.active_count == 0
    assert pending.key not in orchestrator._scheduled_runs
    assert pending.key not in orchestrator._run_tasks


def test_cancel_before_task_start_releases_scheduler_capacity() -> None:
    asyncio.run(_exercise_cancel_before_run_coroutine_starts())


async def _exercise_cancel_refreshes_cleanup_grace(monkeypatch):
    cls = Orchestrator.func_or_class
    orchestrator = cls.__new__(cls)
    old_not_before = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    job = SimpleNamespace(
        id="job",
        run_id="run",
        status="running",
        job_type="magic_clean",
        track_id="track",
        job_options={
            "existing": True,
            "magic_clean_cleanup_tombstone": {
                "b2_key": "prefix/enhanced/job.mp3",
                "bucket_name": "bucket",
                "run_id": "run",
                "not_before": old_not_before,
            },
        },
    )
    track_job = SimpleNamespace(
        status="running",
        current_stage="enhancing",
        completed_at=None,
        updated_at=None,
    )

    class Query:
        def __init__(self, model):
            self.model = model

        def filter(self, *_args):
            return self

        def first(self):
            return job if self.model is not AiTrackJob else track_job

        def update(self, values, **_kwargs):
            target = job if self.model is not AiTrackJob else track_job
            for attribute, value in values.items():
                setattr(target, attribute.key, value)
            return 1

    class Session:
        committed = False
        closed = False

        @staticmethod
        def query(model):
            return Query(model)

        def refresh(self, _job):
            return None

        def rollback(self):
            raise AssertionError("cancellation should commit")

        def close(self):
            self.closed = True

    session = Session()

    async def commit(candidate):
        candidate.committed = True

    cancelled_runs = []
    events = []
    orchestrator._cancel_scheduled_run = (
        lambda job_id, run_id: cancelled_runs.append((job_id, run_id))
    )
    orchestrator._push_event = lambda _job_id, event: events.append(event)
    monkeypatch.setattr("hear.orchestrator.SessionLocal", lambda: session)
    monkeypatch.setattr("hear.orchestrator.commit_with_retry", commit)
    monkeypatch.setattr(settings, "MAGIC_CLEAN_CLEANUP_GRACE_SECONDS", 600)

    assert await orchestrator.cancel("job") is True

    tombstone = job.job_options["magic_clean_cleanup_tombstone"]
    requested_at = datetime.fromisoformat(tombstone["cancellation_requested_at"])
    not_before = datetime.fromisoformat(tombstone["not_before"])
    assert not_before == requested_at + timedelta(seconds=600)
    assert tombstone["not_before"] != old_not_before
    assert job.job_options["existing"] is True
    assert job.status == "cancelled"
    assert cancelled_runs == [("job", "run")]
    assert events[0]["event"] == "job_cancelled"
    assert session.committed and session.closed


def test_cancel_refreshes_provisional_cleanup_writer_grace(monkeypatch) -> None:
    asyncio.run(_exercise_cancel_refreshes_cleanup_grace(monkeypatch))


def _source_metadata():
    return {
        MAGIC_CLEAN_ROOT_URL_KEY: "https://audio.test/root.wav",
        MAGIC_CLEAN_SOURCE_FILE_SHA256_KEY: FILE_HASH,
        MAGIC_CLEAN_SOURCE_PCM_SHA256_KEY: PCM_HASH,
    }


def _completed_candidate(**option_overrides):
    options = {
        MAGIC_CLEAN_ROOT_URL_KEY: "https://audio.test/root.wav",
        MAGIC_CLEAN_SOURCE_FILE_SHA256_KEY: FILE_HASH,
        MAGIC_CLEAN_SOURCE_PCM_SHA256_KEY: PCM_HASH,
        MAGIC_CLEAN_DELIVERED_FILE_SHA256_KEY: DELIVERED_FILE_HASH,
        MAGIC_CLEAN_DELIVERED_PCM_SHA256_KEY: DELIVERED_PCM_HASH,
        MAGIC_CLEAN_ENGINE_REVISION_KEY: settings.MAGIC_CLEAN_ENGINE_REVISION,
        "magic_clean_controls": dict(LEVELS),
        "magic_clean_validated": True,
    }
    options.update(option_overrides)
    return SimpleNamespace(
        id="prior-job",
        job_options=options,
        result_json={
            "enhanced_audio": {"audio_url": "https://audio.test/prior.mp3"},
            "quality": {
                "quality_score": 0.8,
                "snr_db": 12.0,
                "peak_db": -2.0,
                "lufs": -16.0,
                "clipping_detected": False,
            },
        },
    )


def test_reuse_requires_same_root_hashes_controls_engine_and_validation():
    cls = Orchestrator.func_or_class
    candidate = _completed_candidate()

    assert cls._magic_clean_reuse_candidate(
        [candidate],
        _source_metadata(),
        LEVELS,
    ) is candidate
    assert cls._magic_clean_reuse_candidate(
        [_completed_candidate(magic_clean_controls={**LEVELS, "music": 20})],
        _source_metadata(),
        LEVELS,
    ) is None
    assert cls._magic_clean_reuse_candidate(
        [_completed_candidate(magic_clean_validated=False)],
        _source_metadata(),
        LEVELS,
    ) is None


def test_cross_scope_exact_url_and_hash_matches_fail_closed():
    cls = Orchestrator.func_or_class
    candidate = SimpleNamespace(
        result_json={
            "enhanced_audio": {"audio_url": "https://audio.test/foreign.mp3"}
        },
        job_options={
            MAGIC_CLEAN_DELIVERED_FILE_SHA256_KEY: DELIVERED_FILE_HASH,
            MAGIC_CLEAN_DELIVERED_PCM_SHA256_KEY: DELIVERED_PCM_HASH,
        },
    )

    with pytest.raises(MagicCleanLineageError, match="different track scope"):
        cls._reject_cross_scope_magic_clean_match(
            [candidate],
            submitted_url="https://audio.test/foreign.mp3",
        )
    with pytest.raises(MagicCleanLineageError, match="different track scope"):
        cls._reject_cross_scope_magic_clean_match(
            [candidate],
            submitted_pcm_sha256=DELIVERED_PCM_HASH,
        )


def test_remote_result_must_be_bound_to_expected_artifact_and_valid_hashes():
    cls = Orchestrator.func_or_class
    enhancement = {
        "enhanced_url": "https://cdn.test/current.mp3",
        "b2_key": "prefix/enhanced/current.mp3",
        "bucket_name": "bucket",
        "quality_score": 0.8,
        "snr_db": 12.0,
        "peak_db": -2.0,
        "lufs": -16.0,
        "clipping_detected": False,
        "stage_times": {"total": 1.0},
        "source_file_sha256": FILE_HASH,
        "source_pcm_sha256": PCM_HASH,
        "delivered_file_sha256": DELIVERED_FILE_HASH,
        "delivered_pcm_sha256": DELIVERED_PCM_HASH,
        "engine_revision": settings.MAGIC_CLEAN_ENGINE_REVISION,
    }

    assert cls._validate_magic_clean_enhancement(
        enhancement,
        _source_metadata(),
        expected_key="prefix/enhanced/current.mp3",
        expected_bucket="bucket",
        expected_url="https://cdn.test/current.mp3",
    ) is enhancement
    with pytest.raises(RuntimeError, match="unexpected storage key"):
        cls._validate_magic_clean_enhancement(
            {**enhancement, "b2_key": "prefix/enhanced/other.mp3"},
            _source_metadata(),
            expected_key="prefix/enhanced/current.mp3",
            expected_bucket="bucket",
            expected_url="https://cdn.test/current.mp3",
        )


async def _exercise_post_upload_stage_failure(
    monkeypatch,
    failed_stage,
    failure,
    expected_reason,
):
    cls = Orchestrator.func_or_class
    orchestrator = cls.__new__(cls)
    orchestrator._require_magic_clean_storage_lifetime = lambda _storage: None

    async def set_stage(_db, _job, _track_job, stage, _details):
        if stage == failed_stage:
            raise failure
        return True

    orchestrator._set_stage = set_stage
    orchestrator._prepare_magic_clean_source = AsyncMock(
        return_value=("/tmp/nonexistent-root.audio", _source_metadata(), 0.1)
    )
    orchestrator._magic_clean_lineage_jobs = lambda _db, _job: []
    orchestrator._record_magic_clean_cleanup_tombstone = AsyncMock()
    orchestrator._cleanup_magic_clean_artifact = AsyncMock()
    orchestrator._complete = AsyncMock(
        side_effect=AssertionError("cancelled job must not complete")
    )

    enhancement = {
        "enhanced_url": "https://cdn.test/prefix/enhanced/current.mp3",
        "b2_key": "prefix/enhanced/current.mp3",
        "bucket_name": "bucket",
        "quality_score": 0.8,
        "snr_db": 12.0,
        "peak_db": -2.0,
        "lufs": -16.0,
        "clipping_detected": False,
        "stage_times": {"total": 1.0},
        "source_file_sha256": FILE_HASH,
        "source_pcm_sha256": PCM_HASH,
        "delivered_file_sha256": DELIVERED_FILE_HASH,
        "delivered_pcm_sha256": DELIVERED_PCM_HASH,
        "engine_revision": settings.MAGIC_CLEAN_ENGINE_REVISION,
    }

    async def remote_result():
        orchestrator._record_magic_clean_cleanup_tombstone.assert_awaited_once()
        return enhancement

    orchestrator._magic_clean_handle = SimpleNamespace(
        enhance=SimpleNamespace(remote=lambda **_kwargs: remote_result())
    )

    class FakeStorage:
        bucket_name = "bucket"
        context = SimpleNamespace(model_dump=lambda **_kwargs: {})

        @staticmethod
        def key(*_parts):
            return "prefix/enhanced/current.mp3"

        @staticmethod
        def _public_url(_key):
            return "https://cdn.test/prefix/enhanced/current.mp3"

    monkeypatch.setattr("hear.orchestrator.storage_for_job", lambda _job: FakeStorage())
    monkeypatch.setattr("hear.orchestrator.drop_temp_standalone", lambda _path: None)
    job = SimpleNamespace(
        id="current",
        run_id="run",
        attempts=1,
        track_id="track",
        backend_id="backend",
        job_type="magic_clean",
        input_url="https://audio.test/source.wav",
        job_options={},
    )

    with pytest.raises(type(failure)):
        await orchestrator._process_magic_clean(job, SimpleNamespace(), object())

    orchestrator._cleanup_magic_clean_artifact.assert_awaited_once()
    cleanup_call = orchestrator._cleanup_magic_clean_artifact.await_args
    assert cleanup_call.args[3] is enhancement
    assert cleanup_call.kwargs["reason"] == expected_reason
    orchestrator._complete.assert_not_awaited()


@pytest.mark.parametrize("cancelled_stage", ["mixing", "finalizing"])
def test_post_upload_stage_cancellation_cleans_artifact(
    monkeypatch,
    cancelled_stage,
):
    asyncio.run(
        _exercise_post_upload_stage_failure(
            monkeypatch,
            cancelled_stage,
            asyncio.CancelledError(),
            f"cancelled_during_{cancelled_stage}_transition",
        )
    )


@pytest.mark.parametrize("failed_stage", ["mixing", "finalizing"])
def test_post_upload_stage_error_cleans_artifact(monkeypatch, failed_stage):
    asyncio.run(
        _exercise_post_upload_stage_failure(
            monkeypatch,
            failed_stage,
            RuntimeError("stage transition failed"),
            f"{failed_stage}_transition_failed",
        )
    )


async def _exercise_identical_validated_candidate_skip(monkeypatch):
    cls = Orchestrator.func_or_class
    orchestrator = cls.__new__(cls)
    orchestrator._require_magic_clean_storage_lifetime = lambda _storage: None
    orchestrator._set_stage = AsyncMock(return_value=True)
    orchestrator._prepare_magic_clean_source = AsyncMock(
        return_value=("/tmp/nonexistent-root.audio", _source_metadata(), 0.1)
    )
    candidate = _completed_candidate()
    orchestrator._magic_clean_lineage_jobs = lambda _db, _job: [candidate]
    orchestrator._record_magic_clean_cleanup_tombstone = AsyncMock()
    orchestrator._try_reuse_magic_clean_artifact = AsyncMock(
        return_value={
            "enhanced_url": "https://cdn.test/prefix/enhanced/current.mp3",
            "b2_key": "prefix/enhanced/current.mp3",
            "bucket_name": "bucket",
            "quality_score": 0.8,
            "snr_db": 12.0,
            "peak_db": -2.0,
            "lufs": -16.0,
            "clipping_detected": False,
            "stage_times": {"reuse_download": 0.1, "total": 0.2},
            "source_file_sha256": FILE_HASH,
            "source_pcm_sha256": PCM_HASH,
            "delivered_file_sha256": DELIVERED_FILE_HASH,
            "delivered_pcm_sha256": DELIVERED_PCM_HASH,
            "engine_revision": settings.MAGIC_CLEAN_ENGINE_REVISION,
        }
    )
    orchestrator._complete = AsyncMock(return_value=True)
    remote = AsyncMock(side_effect=AssertionError("model inference must be skipped"))
    orchestrator._magic_clean_handle = SimpleNamespace(
        enhance=SimpleNamespace(remote=remote)
    )

    class FakeStorage:
        bucket_name = "bucket"

        @staticmethod
        def key(*_parts):
            return "prefix/enhanced/current.mp3"

        @staticmethod
        def _public_url(_key):
            return "https://cdn.test/prefix/enhanced/current.mp3"

    monkeypatch.setattr("hear.orchestrator.storage_for_job", lambda _job: FakeStorage())
    job = SimpleNamespace(
        id="current",
        run_id="run",
        track_id="track",
        backend_id="backend",
        job_type="magic_clean",
        input_url="https://audio.test/prior.mp3",
        job_options={
            "speech": 100,
            "music": 10,
            "background": 10,
            "cut_silence": False,
            "magic_clean_cleanup_tombstone": {
                "b2_key": "prefix/enhanced/current.mp3"
            },
        },
    )
    track_job = SimpleNamespace()

    await orchestrator._process_magic_clean(job, track_job, object())

    remote.assert_not_awaited()
    assert job.job_options["magic_clean_reused_from_job_id"] == "prior-job"
    assert "magic_clean_cleanup_tombstone" not in job.job_options
    orchestrator._complete.assert_awaited_once()


def test_identical_validated_candidate_skips_remote_model_call(monkeypatch):
    asyncio.run(_exercise_identical_validated_candidate_skip(monkeypatch))


async def _exercise_reuse_upload_cancellation(monkeypatch, tmp_path):
    cls = Orchestrator.func_or_class
    orchestrator = cls.__new__(cls)
    candidate = _completed_candidate()
    reused_path = tmp_path / "reused.mp3"
    reused_path.write_bytes(b"validated-artifact")
    writer_started = threading.Event()
    release_writer = threading.Event()
    upload_finished = threading.Event()
    deleted = []

    async def fake_download(*_args, **_kwargs):
        return str(reused_path)

    class FakeStorage:
        bucket_name = "bucket"

        @staticmethod
        def key(*_parts):
            return "prefix/enhanced/current.mp3"

        @staticmethod
        def upload_file(*_args, **_kwargs):
            writer_started.set()
            assert release_writer.wait(timeout=2)
            upload_finished.set()
            return "https://cdn.test/prefix/enhanced/current.mp3"

        @staticmethod
        def delete_object(key):
            assert upload_finished.is_set()
            deleted.append(key)

    monkeypatch.setattr("hear.orchestrator.download_audio", fake_download)
    monkeypatch.setattr(
        "hear.orchestrator.magic_clean_artifact_hashes",
        lambda _path: (DELIVERED_FILE_HASH, DELIVERED_PCM_HASH),
    )
    monkeypatch.setattr("hear.orchestrator.drop_temp_standalone", lambda _path: None)
    job = SimpleNamespace(
        id="current",
        run_id="run",
        attempts=1,
        track_id="track",
        job_options={},
    )

    task = asyncio.create_task(
        orchestrator._try_reuse_magic_clean_artifact(
            object(),
            job,
            FakeStorage(),
            candidate,
            _source_metadata(),
        )
    )
    assert await asyncio.to_thread(writer_started.wait, 2)
    task.cancel()
    await asyncio.sleep(0.02)
    assert not task.done()
    release_writer.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert deleted == ["prefix/enhanced/current.mp3"]


def test_reuse_upload_finishes_and_is_cleaned_before_cancellation_propagates(
    monkeypatch,
    tmp_path,
):
    asyncio.run(_exercise_reuse_upload_cancellation(monkeypatch, tmp_path))


async def _exercise_reuse_hash_cancellation(monkeypatch, tmp_path):
    cls = Orchestrator.func_or_class
    orchestrator = cls.__new__(cls)
    candidate = _completed_candidate()
    reused_path = tmp_path / "reused.mp3"
    reused_path.write_bytes(b"validated-artifact")
    hash_started = threading.Event()
    release_hash = threading.Event()
    hash_finished = threading.Event()
    cleaned = []

    async def fake_download(*_args, **_kwargs):
        return str(reused_path)

    def blocking_hash(_path):
        hash_started.set()
        assert release_hash.wait(timeout=2)
        hash_finished.set()
        return DELIVERED_FILE_HASH, DELIVERED_PCM_HASH

    class FakeStorage:
        bucket_name = "bucket"

        @staticmethod
        def key(*_parts):
            return "prefix/enhanced/current.mp3"

        @staticmethod
        def upload_file(*_args, **_kwargs):
            raise AssertionError("cancelled hash must not proceed to upload")

    monkeypatch.setattr("hear.orchestrator.download_audio", fake_download)
    monkeypatch.setattr(
        "hear.orchestrator.magic_clean_artifact_hashes",
        blocking_hash,
    )

    def record_cleanup(path):
        assert hash_finished.is_set()
        cleaned.append(path)

    monkeypatch.setattr("hear.orchestrator.drop_temp_standalone", record_cleanup)
    job = SimpleNamespace(
        id="current",
        run_id="run",
        attempts=1,
        track_id="track",
        job_options={},
    )

    task = asyncio.create_task(
        orchestrator._try_reuse_magic_clean_artifact(
            object(),
            job,
            FakeStorage(),
            candidate,
            _source_metadata(),
        )
    )
    assert await asyncio.to_thread(hash_started.wait, 2)
    task.cancel()
    await asyncio.sleep(0.02)
    assert not task.done()
    release_hash.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert cleaned == [str(reused_path)]


def test_reuse_hash_finishes_before_cancellation_cleans_temp_file(
    monkeypatch,
    tmp_path,
):
    asyncio.run(_exercise_reuse_hash_cancellation(monkeypatch, tmp_path))


async def _exercise_deferred_cleanup_tombstone(monkeypatch):
    cls = Orchestrator.func_or_class
    orchestrator = cls.__new__(cls)
    persisted_job = SimpleNamespace(job_options={"existing": True})

    class FakeQuery:
        @staticmethod
        def filter(*_args):
            return FakeQuery()

        @staticmethod
        def with_for_update():
            return FakeQuery()

        @staticmethod
        def first():
            return persisted_job

    class FakeSession:
        committed = False
        closed = False

        @staticmethod
        def query(*_args):
            return FakeQuery()

        @staticmethod
        def rollback():
            raise AssertionError("deferred tombstone should commit")

        def close(self):
            self.closed = True

    cleanup_session = FakeSession()

    async def fake_commit(session):
        session.committed = True

    monkeypatch.setattr("hear.orchestrator.SessionLocal", lambda: cleanup_session)
    monkeypatch.setattr("hear.orchestrator.commit_with_retry", fake_commit)
    job = SimpleNamespace(
        id="current",
        run_id="run",
        attempts=2,
        job_options={},
    )

    def unexpected_delete(_key):
        raise AssertionError("ownership-deferred cleanup must not delete eagerly")

    storage = SimpleNamespace(delete_object=unexpected_delete)

    await orchestrator._cleanup_magic_clean_artifact(
        object(),
        job,
        storage,
        {
            "b2_key": "prefix/enhanced/current.mp3",
            "bucket_name": "bucket",
            "delivered_file_sha256": DELIVERED_FILE_HASH,
        },
        reason="completion_transition_failed",
        defer_deletion=True,
    )

    tombstone = persisted_job.job_options["magic_clean_cleanup_tombstone"]
    assert tombstone["run_id"] == "run"
    assert tombstone["job_attempt"] == 2
    assert tombstone["delivered_file_sha256"] == DELIVERED_FILE_HASH
    assert tombstone["last_error_type"] == "DeferredOwnershipCheck"
    assert tombstone["not_before"] > tombstone["created_at"]
    assert cleanup_session.committed and cleanup_session.closed


def test_completion_uncertainty_records_owned_tombstone_without_eager_delete(
    monkeypatch,
):
    asyncio.run(_exercise_deferred_cleanup_tombstone(monkeypatch))
