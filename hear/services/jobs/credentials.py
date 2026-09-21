"""Attempt-scoped storage credential rotation for durable jobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hear.core.backend_registry import BackendRegistry
from hear.core.storage import StorageContextError, StorageContexts, StorageCredentialsExpiringError
from hear.models.database import AiJob, AiTrackJob, DatabaseRuntime
from hear.models.schemas import StorageContext
from hear.services.jobs.submission import SubmissionConflictError, SubmissionPolicy


@dataclass(frozen=True)
class CredentialRefreshResult:
    job_id: str
    run_id: str
    status: str


class JobCredentialService:
    """Refresh only credentials; payload and destination are immutable."""

    def __init__(self, orchestrator: Any | None = None) -> None:
        self._orchestrator = orchestrator

    async def refresh(
        self, *, backend_id: str, job_id: str, storage: StorageContext
    ) -> CredentialRefreshResult:
        BackendRegistry.validate_storage_for_backend(backend_id, storage)
        db = DatabaseRuntime.SessionLocal()
        try:
            job = (
                db.query(AiJob)
                .filter(AiJob.id == job_id, AiJob.backend_id == backend_id)
                .with_for_update()
                .first()
            )
            if job is None:
                raise LookupError("job not found")
            if job.status != "queued":
                raise ValueError("storage credentials can only be refreshed while a job is queued")
            try:
                previous = StorageContexts.decrypt_storage_context(
                    job.storage_context_encrypted, require_active=False
                )
            except StorageContextError as exc:
                raise ValueError("job has no refreshable storage context") from exc
            if not SubmissionPolicy._storage_destination_matches(previous, storage):
                raise SubmissionConflictError("storage destination cannot be changed for an existing job")
            if job.job_type == "magic_clean":
                SubmissionPolicy._validate_magic_clean_storage_ttl(storage)
            job.storage_context_encrypted = StorageContexts.encrypt_storage_context(storage)
            options = dict(job.job_options or {})
            options["storage_destination"] = {
                key: value
                for key, value in storage.model_dump(mode="json").items()
                if key not in {"key_id", "application_key"}
            }
            job.job_options = options
            if job.error == StorageCredentialsExpiringError.code:
                job.error = None
                db.query(AiTrackJob).filter(
                    AiTrackJob.job_id == job.id,
                    AiTrackJob.run_id == job.run_id,
                    AiTrackJob.status == "queued",
                ).update(
                    {AiTrackJob.error: None}, synchronize_session=False
                )
            db.commit()
            result = CredentialRefreshResult(job_id=job.id, run_id=job.run_id, status=job.status)
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
        if self._orchestrator is not None:
            await self._orchestrator.enqueue.remote(result.job_id, result.run_id)
        return result
