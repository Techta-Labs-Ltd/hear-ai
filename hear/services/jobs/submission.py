"""Atomic, idempotent submission for asynchronous Hear jobs."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ray.serve.handle import DeploymentHandle
from sqlalchemy.dialects.postgresql import insert

from hear.config import settings
from hear.core.backend_registry import BackendRegistry
from hear.core.storage import StorageContextError, StorageContexts, StorageCredentialsExpiringError
from hear.models.database import AiJob, DatabaseRuntime
from hear.models.schemas import PipelineRequest, StorageContext
from hear.services.magic_clean.models import DEFAULT_STEM_LEVELS

ALLOWED_JOB_TYPES = {
    "pipeline",
    "magic_clean",
    "transcription",
    "categorization",
    "audio_tag",
    "rebuild",
    "reconstruct",
    "edit_transcript",
    "discovery",
}
# Compatibility names are normalized before validation and persisted as the
# canonical workflow.  Unknown names remain explicit validation failures.
JOB_TYPE_ALIASES = {"tagging": "categorization"}
AUDIO_REQUIRED_JOB_TYPES = {
    "pipeline",
    "magic_clean",
    "transcription",
    "audio_tag",
    "reconstruct",
    "edit_transcript",
    "discovery",
}
MAGIC_CLEAN_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


class SubmissionConflictError(Exception):
    """The idempotency key was already used with a different request."""


class SubmissionUnavailableError(Exception):
    """The job is durable but Ray could not acknowledge dispatch."""


@dataclass(frozen=True)
class SubmissionResult:
    backend_id: str
    job_id: str
    run_id: str
    track_id: str
    job_type: str
    status: str
    replayed: bool


class SubmissionPolicy:
    @staticmethod
    def _magic_clean_storage_ttl_is_safe(storage: StorageContext) -> bool:
        required_storage_ttl = settings.MAGIC_CLEAN_STORAGE_CREDENTIAL_MIN_TTL_SECONDS
        if not math.isfinite(required_storage_ttl) or required_storage_ttl <= 0:
            raise RuntimeError("MAGIC_CLEAN_STORAGE_CREDENTIAL_MIN_TTL_SECONDS is invalid")
        remaining_storage_ttl = (storage.expires_at - datetime.now(UTC)).total_seconds()
        return remaining_storage_ttl >= required_storage_ttl

    @staticmethod
    def _validate_magic_clean_storage_ttl(storage: StorageContext) -> None:
        if not SubmissionPolicy._magic_clean_storage_ttl_is_safe(storage):
            required_storage_ttl = settings.MAGIC_CLEAN_STORAGE_CREDENTIAL_MIN_TTL_SECONDS
            raise StorageCredentialsExpiringError(
                f"magic_clean storage credentials must remain valid for at least {required_storage_ttl:g} seconds"
            )

    @staticmethod
    def _credential_material_matches(stored: StorageContext, submitted: StorageContext) -> bool:
        return (
            stored.key_id == submitted.key_id
            and stored.application_key == submitted.application_key
        )

    @staticmethod
    def _storage_destination_matches(stored: StorageContext, submitted: StorageContext) -> bool:
        return all(
            getattr(stored, field) == getattr(submitted, field)
            for field in ("endpoint_url", "bucket_name", "folder_prefix", "public_base_url")
        )

    @staticmethod
    def _has_valid_magic_clean_cleanup_tombstone(job: AiJob, storage: StorageContext) -> bool:
        raw_options = job.job_options
        options: dict[str, Any] = raw_options if isinstance(raw_options, dict) else {}
        tombstone = options.get("magic_clean_cleanup_tombstone")
        if not isinstance(tombstone, dict):
            return False
        key = str(tombstone.get("b2_key") or "").strip()
        bucket_name = str(tombstone.get("bucket_name") or "").strip()
        tombstone_run_id = str(tombstone.get("run_id") or "").strip()
        job_run_id = str(job.run_id or "").strip()
        if (
            not key
            or not job_run_id
            or tombstone_run_id != job_run_id
            or (bucket_name != storage.bucket_name)
            or (not key.startswith(storage.folder_prefix))
            or ("\\" in key)
            or ("\x00" in key)
        ):
            return False
        relative_key = key.removeprefix(storage.folder_prefix)
        return bool(relative_key) and all(
            part not in {"", ".", ".."} for part in relative_key.split("/")
        )

    @staticmethod
    def normalize_request(
        request: PipelineRequest, *, enforce_magic_clean_storage_ttl: bool = True
    ) -> dict[str, Any]:
        job_id = request.job_id.strip()
        track_id = request.track_id.strip()
        job_type = (request.job_type or "pipeline").strip().replace("-", "_")
        job_type = JOB_TYPE_ALIASES.get(job_type, job_type)
        user_id = request.user_id.strip()
        backend_id = request.backend_id.strip()
        if not job_id or not track_id:
            raise ValueError("job_id and track_id are required")
        if not user_id:
            raise ValueError("user_id is required")
        if not backend_id:
            raise ValueError("backend_id is required")
        BackendRegistry.validate_storage_for_backend(backend_id, request.storage)
        if job_type not in ALLOWED_JOB_TYPES:
            raise ValueError(f"unsupported job_type: {request.job_type}")
        if job_type == "magic_clean" and enforce_magic_clean_storage_ttl:
            SubmissionPolicy._validate_magic_clean_storage_ttl(request.storage)
        if job_type in {"rebuild", "edit_transcript"} and (
            not (request.edited_transcript or "").strip()
        ):
            raise ValueError(f"edited_transcript is required for {job_type}")
        if job_type == "reconstruct" and (not request.changes):
            raise ValueError("changes are required for reconstruct")
        if job_type in AUDIO_REQUIRED_JOB_TYPES and (not (request.audio_url or "").strip()):
            raise ValueError(f"audio_url is required for {job_type}")
        if job_type == "categorization" and (
            not ((request.audio_url or "").strip() or (request.edited_transcript or "").strip())
        ):
            raise ValueError("audio_url or edited_transcript is required for categorization")
        changes = []
        for change in request.changes:
            if change.segment_end <= change.segment_start:
                raise ValueError("each change must end after it starts")
            if not change.new_text.strip():
                raise ValueError("each change requires new_text")
            changes.append(
                {
                    "segment_start": change.segment_start,
                    "segment_end": change.segment_end,
                    "new_text": change.new_text,
                    "original_text": change.original_text,
                }
            )
        magic_clean_controls_omitted = job_type == "magic_clean" and all(
            value is None for value in (request.speech, request.music, request.background)
        )
        speech: int | None
        music: int | None
        background: int | None
        if magic_clean_controls_omitted:
            speech = DEFAULT_STEM_LEVELS.speech
            music = DEFAULT_STEM_LEVELS.music
            background = DEFAULT_STEM_LEVELS.background
        else:
            speech = request.speech
            music = request.music
            background = request.background
        return {
            "backend_id": backend_id,
            "storage": request.storage.model_dump(mode="json"),
            "job_id": job_id,
            "track_id": track_id,
            "job_type": job_type,
            "max_tags": request.max_tags or 8,
            "audio_url": request.audio_url,
            "edited_transcript": request.edited_transcript,
            "changes": changes,
            "same_speaker": request.same_speaker,
            "grouped": request.grouped,
            "group_id": request.group_id,
            "kind": request.kind or "track",
            "source": request.source,
            "track_count": request.track_count or 1,
            "playback_instruction": request.playback_instruction,
            "type": request.type,
            "media_file_id": request.media_file_id,
            "user_id": user_id,
            "speech": speech,
            "music": music,
            "background": background,
            "cut_silence": request.cut_silence,
            "magic_clean_controls_omitted": magic_clean_controls_omitted,
        }

    @staticmethod
    def request_fingerprint(payload: dict[str, Any]) -> str:
        """Fingerprint immutable job semantics, never expiring credentials."""
        return SubmissionPolicy._request_fingerprint(payload, ignore_storage_expiration=True)

    @staticmethod
    def _semantic_request_fingerprint(payload: dict[str, Any]) -> str:
        """Ignore rotatable credentials while retaining the storage destination."""
        return SubmissionPolicy._request_fingerprint(payload, ignore_storage_expiration=True)

    @staticmethod
    def _request_fingerprint(payload: dict[str, Any], *, ignore_storage_expiration: bool) -> str:
        canonical = {
            key: value
            for key, value in payload.items()
            if key not in {"job_id", "storage", "magic_clean_controls_omitted"}
        }
        ignored_storage_fields = {"key_id", "application_key"}
        if ignore_storage_expiration:
            ignored_storage_fields.add("expires_at")
        canonical["storage"] = {
            key: value
            for key, value in payload["storage"].items()
            if key not in ignored_storage_fields
        }
        encoded = json.dumps(
            canonical, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _job_options(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "backend_id": payload["backend_id"],
            "storage_destination": {
                key: value
                for key, value in payload["storage"].items()
                if key not in {"key_id", "application_key"}
            },
            "grouped": payload["grouped"],
            "group_id": payload["group_id"],
            "kind": payload["kind"],
            "source": payload["source"],
            "track_count": payload["track_count"],
            "playback_instruction": payload["playback_instruction"],
            "type": payload["type"],
            "media_file_id": payload["media_file_id"],
            "user_id": payload["user_id"],
            "same_speaker": payload["same_speaker"],
            "speech": payload["speech"],
            "music": payload["music"],
            "background": payload["background"],
            "cut_silence": payload["cut_silence"],
            "magic_clean_controls_omitted": payload["magic_clean_controls_omitted"],
        }

    @staticmethod
    def _reconstruct_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
        if not payload["changes"]:
            return None
        return {"changes": payload["changes"], "same_speaker": payload["same_speaker"]}

    @staticmethod
    def _legacy_payload(job: AiJob) -> dict[str, Any]:
        raw_options = job.job_options
        options: dict[str, Any] = raw_options if isinstance(raw_options, dict) else {}
        raw_reconstruct = job.custom_tags
        reconstruct: dict[str, Any] = raw_reconstruct if isinstance(raw_reconstruct, dict) else {}
        return {
            "backend_id": job.backend_id or "",
            "storage": options.get("storage_destination") or {},
            "job_id": job.id,
            "track_id": job.track_id or "",
            "job_type": (job.job_type or "pipeline").replace("-", "_"),
            "max_tags": job.max_tags or 8,
            "audio_url": job.input_url,
            "edited_transcript": job.edited_transcript,
            "changes": reconstruct.get("changes") or [],
            "same_speaker": reconstruct.get("same_speaker", options.get("same_speaker", True)),
            "grouped": options.get("grouped", False),
            "group_id": options.get("group_id"),
            "kind": options.get("kind") or "track",
            "source": options.get("source"),
            "track_count": options.get("track_count") or 1,
            "playback_instruction": options.get("playback_instruction"),
            "type": options.get("type"),
            "media_file_id": options.get("media_file_id"),
            "user_id": options.get("user_id"),
            "speech": options.get("speech"),
            "music": options.get("music"),
            "background": options.get("background"),
            "cut_silence": bool(options.get("cut_silence", False)),
            "magic_clean_controls_omitted": bool(
                options.get("magic_clean_controls_omitted", False)
            ),
        }


class JobSubmissionService:
    def __init__(self, orchestrator: DeploymentHandle) -> None:
        self._orchestrator = orchestrator

    async def submit(self, request: PipelineRequest) -> SubmissionResult:
        payload = SubmissionPolicy.normalize_request(request, enforce_magic_clean_storage_ttl=False)
        fingerprint = SubmissionPolicy.request_fingerprint(payload)
        run_id = str(uuid.uuid4())
        now = datetime.now(UTC).replace(tzinfo=None)
        values = {
            "id": payload["job_id"],
            "backend_id": payload["backend_id"],
            "storage_context_encrypted": StorageContexts.encrypt_storage_context(request.storage),
            "run_id": run_id,
            "job_type": payload["job_type"],
            "track_id": payload["track_id"],
            "status": "queued",
            "current_stage": None,
            "input_url": payload["audio_url"],
            "max_tags": payload["max_tags"],
            "edited_transcript": payload["edited_transcript"],
            "custom_tags": SubmissionPolicy._reconstruct_payload(payload),
            "job_options": SubmissionPolicy._job_options(payload),
            "request_hash": fingerprint,
            "attempts": 0,
            "created_at": now,
        }
        db = DatabaseRuntime.SessionLocal()
        try:
            statement = (
                insert(AiJob)
                .values(**values)
                .on_conflict_do_nothing(index_elements=[AiJob.id])
                .returning(AiJob.id)
            )
            inserted = db.execute(statement).scalar_one_or_none() is not None
            if inserted and payload["job_type"] == "magic_clean":
                SubmissionPolicy._validate_magic_clean_storage_ttl(request.storage)
            db.commit()
            job = (
                db.query(AiJob)
                .filter(AiJob.id == payload["job_id"])
                .populate_existing()
                .with_for_update()
                .first()
            )
            if job is None:
                raise RuntimeError("job disappeared after submission")
            parked_for_storage_refresh = False
            if not inserted:
                stored_fingerprint = job.request_hash
                if not stored_fingerprint:
                    stored_fingerprint = SubmissionPolicy.request_fingerprint(
                        SubmissionPolicy._legacy_payload(job)
                    )
                    if stored_fingerprint == fingerprint:
                        job.request_hash = fingerprint
                exact_payload_fingerprint = stored_fingerprint == fingerprint
                if not exact_payload_fingerprint:
                    stored_semantic_fingerprint = SubmissionPolicy._semantic_request_fingerprint(
                        SubmissionPolicy._legacy_payload(job)
                    )
                    if stored_semantic_fingerprint != SubmissionPolicy._semantic_request_fingerprint(
                        payload
                    ):
                        raise SubmissionConflictError(
                            "job_id has already been used with a different payload"
                        )
                # Every queued job accepts a credential rotation only when it
                # targets the same immutable storage destination.  Magic Clean
                # adds its longer cleanup-window checks below.
                if payload["job_type"] != "magic_clean":
                    try:
                        stored_storage = StorageContexts.decrypt_storage_context(
                            getattr(job, "storage_context_encrypted", None), require_active=False
                        )
                    except StorageContextError as exc:
                        raise ValueError("job has no refreshable storage context") from exc
                    if not SubmissionPolicy._storage_destination_matches(
                        stored_storage, request.storage
                    ):
                        raise SubmissionConflictError(
                            "storage destination cannot be changed for an existing job"
                        )
                    if job.status != "queued" and (
                        not SubmissionPolicy._credential_material_matches(
                            stored_storage, request.storage
                        )
                        or request.storage.expires_at != stored_storage.expires_at
                    ):
                        raise ValueError("storage credentials cannot be refreshed after the job has started")
                    if job.status == "queued":
                        job.storage_context_encrypted = values["storage_context_encrypted"]
                        refreshed_options = dict(job.job_options or {})
                        refreshed_options["storage_destination"] = {
                            key: value
                            for key, value in payload["storage"].items()
                            if key not in {"key_id", "application_key"}
                        }
                        job.job_options = refreshed_options
                        job.request_hash = fingerprint
                if payload["job_type"] == "magic_clean" and (
                    job.status in {"queued", "running"}
                    or job.status in MAGIC_CLEAN_TERMINAL_STATUSES
                ):
                    try:
                        stored_storage = StorageContexts.decrypt_storage_context(
                            getattr(job, "storage_context_encrypted", None), require_active=False
                        )
                    except StorageContextError:
                        stored_storage = None
                    credential_material_changed = (
                        stored_storage is None
                        or not SubmissionPolicy._credential_material_matches(
                            stored_storage, request.storage
                        )
                    )
                    expiration_advances = (
                        stored_storage is None
                        or request.storage.expires_at > stored_storage.expires_at
                    )
                    expiration_matches = (
                        stored_storage is not None
                        and request.storage.expires_at == stored_storage.expires_at
                    )
                    exact_credential_replay = (
                        exact_payload_fingerprint
                        and stored_storage is not None
                        and (not credential_material_changed)
                        and expiration_matches
                    )
                    refresh_storage = False
                    if job.status in MAGIC_CLEAN_TERMINAL_STATUSES:
                        if stored_storage is None:
                            if not exact_payload_fingerprint:
                                raise ValueError(
                                    "terminal magic_clean storage credentials cannot be renewed without a valid stored context"
                                )
                        elif not exact_credential_replay:
                            if not SubmissionPolicy._storage_destination_matches(
                                stored_storage, request.storage
                            ):
                                raise SubmissionConflictError(
                                    "magic_clean storage destination cannot be changed"
                                )
                            if not SubmissionPolicy._has_valid_magic_clean_cleanup_tombstone(
                                job, stored_storage
                            ):
                                raise ValueError(
                                    "terminal magic_clean credential renewal requires a valid cleanup tombstone"
                                )
                            if not expiration_advances:
                                raise ValueError(
                                    "terminal magic_clean credential renewal must use a later expiration"
                                )
                            SubmissionPolicy._validate_magic_clean_storage_ttl(request.storage)
                            refresh_storage = True
                    elif job.status == "running":
                        if (
                            stored_storage is None
                            or credential_material_changed
                            or expiration_advances
                        ):
                            raise ValueError(
                                "magic_clean storage credentials cannot be refreshed after the job has started"
                            )
                    elif credential_material_changed and (not expiration_advances):
                        raise ValueError(
                            "magic_clean credential rotation must use a later expiration"
                        )
                    elif expiration_advances:
                        SubmissionPolicy._validate_magic_clean_storage_ttl(request.storage)
                        refresh_storage = True
                    if refresh_storage:
                        job.storage_context_encrypted = values["storage_context_encrypted"]
                        refreshed_options = dict(job.job_options or {})
                        refreshed_options["storage_destination"] = {
                            key: value
                            for key, value in payload["storage"].items()
                            if key not in {"key_id", "application_key"}
                        }
                        job.job_options = refreshed_options
                        job.request_hash = fingerprint
                        stored_storage = request.storage
                    if job.status == "queued" and (
                        stored_storage is None
                        or not SubmissionPolicy._magic_clean_storage_ttl_is_safe(stored_storage)
                    ):
                        if exact_credential_replay:
                            job.error = StorageCredentialsExpiringError.code
                            parked_for_storage_refresh = True
                        else:
                            SubmissionPolicy._validate_magic_clean_storage_ttl(request.storage)
                            raise RuntimeError(
                                "queued magic_clean job has no cleanup-safe storage context"
                            )
                    elif (
                        job.status == "queued" and job.error == StorageCredentialsExpiringError.code
                    ):
                        job.error = None
            should_enqueue = job.status == "queued" and (not parked_for_storage_refresh)
            db.commit()
            if should_enqueue:
                try:
                    await self._orchestrator.enqueue.remote(job.id, job.run_id)
                except Exception as exc:
                    raise SubmissionUnavailableError(
                        "job was saved but Ray dispatch was not acknowledged; retry the same job_id"
                    ) from exc
            return SubmissionResult(
                backend_id=job.backend_id,
                job_id=job.id,
                run_id=job.run_id,
                track_id=job.track_id or "",
                job_type=job.job_type or "pipeline",
                status=job.status or "queued",
                replayed=not inserted,
            )
        except (SubmissionConflictError, SubmissionUnavailableError):
            db.rollback()
            raise
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
