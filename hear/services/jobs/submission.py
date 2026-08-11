"""Atomic, idempotent submission for asynchronous Hear jobs."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ray.serve.handle import DeploymentHandle
from sqlalchemy.dialects.postgresql import insert

from hear.models.database import AiJob, SessionLocal
from hear.core.backend_registry import validate_storage_for_backend
from hear.core.storage import encrypt_storage_context
from hear.models.schemas import PipelineRequest
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

AUDIO_REQUIRED_JOB_TYPES = {
    "pipeline",
    "magic_clean",
    "transcription",
    "audio_tag",
    "reconstruct",
    "edit_transcript",
    "discovery",
}


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


def normalize_request(request: PipelineRequest) -> dict[str, Any]:
    job_id = request.job_id.strip()
    track_id = request.track_id.strip()
    job_type = (request.job_type or "pipeline").strip().replace("-", "_")
    user_id = request.user_id.strip()
    backend_id = request.backend_id.strip()
    if not job_id or not track_id:
        raise ValueError("job_id and track_id are required")
    if not user_id:
        raise ValueError("user_id is required")
    if not backend_id:
        raise ValueError("backend_id is required")
    validate_storage_for_backend(backend_id, request.storage)
    if job_type not in ALLOWED_JOB_TYPES:
        raise ValueError(f"unsupported job_type: {request.job_type}")
    if job_type in {"rebuild", "edit_transcript"} and not (
        request.edited_transcript or ""
    ).strip():
        raise ValueError(f"edited_transcript is required for {job_type}")
    if job_type == "reconstruct" and not request.changes:
        raise ValueError("changes are required for reconstruct")
    if job_type in AUDIO_REQUIRED_JOB_TYPES and not (request.audio_url or "").strip():
        raise ValueError(f"audio_url is required for {job_type}")
    if job_type == "categorization" and not (
        (request.audio_url or "").strip()
        or (request.edited_transcript or "").strip()
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

    if job_type == "magic_clean" and request.speech is None:
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
        "speed_multipliers": list(request.speed_multipliers or []),
        "playback_instruction": request.playback_instruction,
        "type": request.type,
        "media_file_id": request.media_file_id,
        "user_id": user_id,
        "speech": speech,
        "music": music,
        "background": background,
        "cut_silence": request.cut_silence,
    }


def request_fingerprint(payload: dict[str, Any]) -> str:
    canonical = {
        key: value
        for key, value in payload.items()
        if key not in {"job_id", "storage"}
    }
    canonical["storage"] = {
        key: value
        for key, value in payload["storage"].items()
        if key not in {"key_id", "application_key"}
    }
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
        "speed_multipliers": payload["speed_multipliers"],
        "playback_instruction": payload["playback_instruction"],
        "type": payload["type"],
        "media_file_id": payload["media_file_id"],
        "user_id": payload["user_id"],
        "same_speaker": payload["same_speaker"],
        "speech": payload["speech"],
        "music": payload["music"],
        "background": payload["background"],
        "cut_silence": payload["cut_silence"],
    }


def _reconstruct_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    if not payload["changes"]:
        return None
    return {
        "changes": payload["changes"],
        "same_speaker": payload["same_speaker"],
    }


def _legacy_payload(job: AiJob) -> dict[str, Any]:
    options = job.job_options or {}
    reconstruct = job.custom_tags or {}
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
        "same_speaker": reconstruct.get(
            "same_speaker", options.get("same_speaker", True)
        ),
        "grouped": options.get("grouped", False),
        "group_id": options.get("group_id"),
        "kind": options.get("kind") or "track",
        "source": options.get("source"),
        "track_count": options.get("track_count") or 1,
        "speed_multipliers": list(options.get("speed_multipliers") or []),
        "playback_instruction": options.get("playback_instruction"),
        "type": options.get("type"),
        "media_file_id": options.get("media_file_id"),
        "user_id": options.get("user_id"),
        "speech": options.get("speech"),
        "music": options.get("music"),
        "background": options.get("background"),
        "cut_silence": bool(options.get("cut_silence", False)),
    }


class JobSubmissionService:
    def __init__(self, orchestrator: DeploymentHandle) -> None:
        self._orchestrator = orchestrator

    async def submit(self, request: PipelineRequest) -> SubmissionResult:
        payload = normalize_request(request)
        fingerprint = request_fingerprint(payload)
        run_id = str(uuid.uuid4())
        now = datetime.utcnow()
        values = {
            "id": payload["job_id"],
            "backend_id": payload["backend_id"],
            "storage_context_encrypted": encrypt_storage_context(request.storage),
            "run_id": run_id,
            "job_type": payload["job_type"],
            "track_id": payload["track_id"],
            "status": "queued",
            "current_stage": None,
            "input_url": payload["audio_url"],
            "max_tags": payload["max_tags"],
            "edited_transcript": payload["edited_transcript"],
            "custom_tags": _reconstruct_payload(payload),
            "job_options": _job_options(payload),
            "request_hash": fingerprint,
            "attempts": 0,
            "created_at": now,
        }

        db = SessionLocal()
        try:
            statement = (
                insert(AiJob)
                .values(**values)
                .on_conflict_do_nothing(index_elements=[AiJob.id])
                .returning(AiJob.id)
            )
            inserted = db.execute(statement).scalar_one_or_none() is not None
            db.commit()

            job = db.query(AiJob).filter(AiJob.id == payload["job_id"]).first()
            if job is None:
                raise RuntimeError("job disappeared after submission")

            if not inserted:
                stored_fingerprint = job.request_hash
                if not stored_fingerprint:
                    stored_fingerprint = request_fingerprint(_legacy_payload(job))
                    if stored_fingerprint == fingerprint:
                        job.request_hash = fingerprint
                        db.commit()
                if stored_fingerprint != fingerprint:
                    raise SubmissionConflictError(
                        "job_id has already been used with a different payload"
                    )

            if inserted or job.status == "queued":
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
