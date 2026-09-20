"""Durable reconciliation for uncommitted Magic Clean storage artifacts."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from hear.core.storage import StorageContexts
from hear.models.database import AiJob, DatabaseRuntime

logger = logging.getLogger(__name__)
ACTIVE_JOB_STATUSES = {"queued", "running"}
TERMINAL_JOB_STATUSES = {"completed", "failed", "cancelled"}


class MagicCleanCleanup:
    @staticmethod
    def _result_artifact_key(result: object) -> str:
        if not isinstance(result, dict):
            return ""
        enhanced_audio = result.get("enhanced_audio")
        if not isinstance(enhanced_audio, dict):
            return ""
        return str(enhanced_audio.get("b2_key") or "").strip()

    @staticmethod
    def _updated_failed_tombstone(tombstone: dict, *, now: str, error_type: str) -> dict:
        return {
            **tombstone,
            "attempts": int(tombstone.get("attempts") or 0) + 1,
            "last_attempt_at": now,
            "last_error_type": error_type,
        }

    @staticmethod
    def reconcile_magic_clean_cleanup_tombstones() -> dict[str, int]:
        """Retry verified deletion without discarding a failed tombstone."""
        scanned = 0
        deleted = 0
        failed = 0
        db = DatabaseRuntime.SessionLocal()
        try:
            jobs = (
                db.query(AiJob)
                .filter(
                    AiJob.job_options.isnot(None),
                    AiJob.job_options["magic_clean_cleanup_tombstone"].isnot(None),
                )
                .with_for_update(skip_locked=True)
                .all()
            )
            for job in jobs:
                options = job.job_options if isinstance(job.job_options, dict) else {}
                tombstone = options.get("magic_clean_cleanup_tombstone")
                if not isinstance(tombstone, dict):
                    continue
                key = tombstone.get("b2_key")
                if not isinstance(key, str) or not key.strip():
                    continue
                scanned += 1
                now_datetime = datetime.now(UTC)
                now = now_datetime.isoformat()
                status = str(getattr(job, "status", "") or "").strip().lower()
                tombstone_run_id = str(tombstone.get("run_id") or "").strip()
                current_run_id = str(getattr(job, "run_id", "") or "").strip()
                not_before_value = tombstone.get("not_before")
                if isinstance(not_before_value, str) and not_before_value.strip():
                    try:
                        not_before = datetime.fromisoformat(not_before_value)
                        if not_before.tzinfo is None:
                            raise ValueError("cleanup timestamp must include a timezone")
                    except ValueError:
                        failed += 1
                        job.job_options = {
                            **options,
                            "magic_clean_cleanup_tombstone": MagicCleanCleanup._updated_failed_tombstone(
                                tombstone, now=now, error_type="InvalidCleanupNotBefore"
                            ),
                        }
                        continue
                    if now_datetime < not_before:
                        continue
                if status in ACTIVE_JOB_STATUSES:
                    continue
                if tombstone_run_id and current_run_id and (tombstone_run_id != current_run_id):
                    failed += 1
                    job.job_options = {
                        **options,
                        "magic_clean_cleanup_tombstone": MagicCleanCleanup._updated_failed_tombstone(
                            tombstone, now=now, error_type="ArtifactOwnershipMismatch"
                        ),
                    }
                    continue
                if status == "completed":
                    authoritative_key = MagicCleanCleanup._result_artifact_key(
                        getattr(job, "result_json", None)
                    )
                    if authoritative_key == key:
                        updated_options = dict(options)
                        updated_options.pop("magic_clean_cleanup_tombstone", None)
                        updated_options["magic_clean_cleanup_superseded_at"] = now
                        job.job_options = updated_options
                        continue
                    if not authoritative_key:
                        failed += 1
                        job.job_options = {
                            **options,
                            "magic_clean_cleanup_tombstone": MagicCleanCleanup._updated_failed_tombstone(
                                tombstone, now=now, error_type="CompletedArtifactUnknown"
                            ),
                        }
                        continue
                if status not in TERMINAL_JOB_STATUSES:
                    continue
                try:
                    storage = StorageContexts.storage_for_job(job)
                    tombstone_bucket = str(tombstone.get("bucket_name") or "").strip()
                    if tombstone_bucket and tombstone_bucket != storage.bucket_name:
                        raise RuntimeError("cleanup tombstone bucket ownership mismatch")
                    storage.delete_object(key)
                except Exception as exc:
                    failed += 1
                    job.job_options = {
                        **options,
                        "magic_clean_cleanup_tombstone": MagicCleanCleanup._updated_failed_tombstone(
                            tombstone, now=now, error_type=type(exc).__name__
                        ),
                    }
                    logger.error("Magic Clean cleanup reconciliation failed for job=%s", job.id)
                else:
                    deleted += 1
                    updated_options = dict(options)
                    updated_options.pop("magic_clean_cleanup_tombstone", None)
                    updated_options["magic_clean_cleanup_reconciled_at"] = now
                    job.job_options = updated_options
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
        return {"scanned": scanned, "deleted": deleted, "failed": failed}
