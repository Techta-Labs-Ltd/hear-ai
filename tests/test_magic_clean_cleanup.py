from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from hear.services.magic_clean.cleanup import MagicCleanCleanup


class FakeQuery:
    def __init__(self, jobs):
        self.jobs = jobs

    def filter(self, *_args):
        return self

    def with_for_update(self, **_kwargs):
        return self

    def all(self):
        return self.jobs


class FakeSession:
    def __init__(self, jobs):
        self.jobs = jobs
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def query(self, *_args):
        return FakeQuery(self.jobs)

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


def _job(*, status="cancelled", result_json=None, run_id="run-1"):
    return SimpleNamespace(
        id="job-1",
        run_id=run_id,
        status=status,
        result_json=result_json,
        job_options={
            "magic_clean_cleanup_tombstone": {
                "b2_key": "prefix/enhanced/job-1.mp3",
                "bucket_name": "bucket",
                "run_id": "run-1",
                "reason": "completion_lost_to_cancellation",
            }
        },
    )


def test_cleanup_reconciler_removes_verified_tombstone(monkeypatch):
    job = _job()
    session = FakeSession([job])
    deleted = []
    storage = SimpleNamespace(bucket_name="bucket", delete_object=lambda key: deleted.append(key))
    monkeypatch.setattr(
        "hear.services.magic_clean.cleanup.DatabaseRuntime.SessionLocal", lambda: session
    )
    monkeypatch.setattr(
        "hear.services.magic_clean.cleanup.StorageContexts.storage_for_job", lambda _job: storage
    )
    result = MagicCleanCleanup.reconcile_magic_clean_cleanup_tombstones()
    assert result == {"scanned": 1, "deleted": 1, "failed": 0}
    assert deleted == ["prefix/enhanced/job-1.mp3"]
    assert "magic_clean_cleanup_tombstone" not in job.job_options
    assert "magic_clean_cleanup_reconciled_at" in job.job_options
    assert session.committed and session.closed and (not session.rolled_back)


def test_cleanup_reconciler_keeps_and_updates_failed_tombstone(monkeypatch):
    job = _job()
    session = FakeSession([job])

    def fail_delete(_key):
        raise RuntimeError("storage unavailable")

    monkeypatch.setattr(
        "hear.services.magic_clean.cleanup.DatabaseRuntime.SessionLocal", lambda: session
    )
    monkeypatch.setattr(
        "hear.services.magic_clean.cleanup.StorageContexts.storage_for_job",
        lambda _job: SimpleNamespace(bucket_name="bucket", delete_object=fail_delete),
    )
    result = MagicCleanCleanup.reconcile_magic_clean_cleanup_tombstones()
    assert result == {"scanned": 1, "deleted": 0, "failed": 1}
    tombstone = job.job_options["magic_clean_cleanup_tombstone"]
    assert tombstone["attempts"] == 1
    assert tombstone["last_error_type"] == "RuntimeError"
    assert session.committed and session.closed and (not session.rolled_back)


def test_cleanup_reconciler_never_deletes_active_retry_key(monkeypatch):
    job = _job(status="running")
    session = FakeSession([job])
    deleted = []
    monkeypatch.setattr(
        "hear.services.magic_clean.cleanup.DatabaseRuntime.SessionLocal", lambda: session
    )
    monkeypatch.setattr(
        "hear.services.magic_clean.cleanup.StorageContexts.storage_for_job",
        lambda _job: SimpleNamespace(
            bucket_name="bucket", delete_object=lambda key: deleted.append(key)
        ),
    )
    result = MagicCleanCleanup.reconcile_magic_clean_cleanup_tombstones()
    assert result == {"scanned": 1, "deleted": 0, "failed": 0}
    assert not deleted
    assert "magic_clean_cleanup_tombstone" in job.job_options


def test_cleanup_reconciler_waits_for_writer_grace_period(monkeypatch):
    job = _job()
    job.job_options["magic_clean_cleanup_tombstone"]["not_before"] = (
        datetime.now(UTC) + timedelta(minutes=5)
    ).isoformat()
    session = FakeSession([job])
    deleted = []
    monkeypatch.setattr(
        "hear.services.magic_clean.cleanup.DatabaseRuntime.SessionLocal", lambda: session
    )
    monkeypatch.setattr(
        "hear.services.magic_clean.cleanup.StorageContexts.storage_for_job",
        lambda _job: SimpleNamespace(
            bucket_name="bucket", delete_object=lambda key: deleted.append(key)
        ),
    )
    result = MagicCleanCleanup.reconcile_magic_clean_cleanup_tombstones()
    assert result == {"scanned": 1, "deleted": 0, "failed": 0}
    assert not deleted
    assert "magic_clean_cleanup_tombstone" in job.job_options


def test_cleanup_reconciler_preserves_completed_authoritative_key(monkeypatch):
    key = "prefix/enhanced/job-1.mp3"
    job = _job(status="completed", result_json={"enhanced_audio": {"b2_key": key}})
    session = FakeSession([job])
    deleted = []
    monkeypatch.setattr(
        "hear.services.magic_clean.cleanup.DatabaseRuntime.SessionLocal", lambda: session
    )
    monkeypatch.setattr(
        "hear.services.magic_clean.cleanup.StorageContexts.storage_for_job",
        lambda _job: SimpleNamespace(
            bucket_name="bucket", delete_object=lambda candidate: deleted.append(candidate)
        ),
    )
    result = MagicCleanCleanup.reconcile_magic_clean_cleanup_tombstones()
    assert result == {"scanned": 1, "deleted": 0, "failed": 0}
    assert not deleted
    assert "magic_clean_cleanup_tombstone" not in job.job_options
    assert "magic_clean_cleanup_superseded_at" in job.job_options


def test_cleanup_reconciler_rejects_different_run_owner(monkeypatch):
    job = _job(status="failed", run_id="run-2")
    session = FakeSession([job])
    deleted = []
    monkeypatch.setattr(
        "hear.services.magic_clean.cleanup.DatabaseRuntime.SessionLocal", lambda: session
    )
    monkeypatch.setattr(
        "hear.services.magic_clean.cleanup.StorageContexts.storage_for_job",
        lambda _job: SimpleNamespace(
            bucket_name="bucket", delete_object=lambda key: deleted.append(key)
        ),
    )
    result = MagicCleanCleanup.reconcile_magic_clean_cleanup_tombstones()
    assert result == {"scanned": 1, "deleted": 0, "failed": 1}
    assert not deleted
    tombstone = job.job_options["magic_clean_cleanup_tombstone"]
    assert tombstone["last_error_type"] == "ArtifactOwnershipMismatch"
