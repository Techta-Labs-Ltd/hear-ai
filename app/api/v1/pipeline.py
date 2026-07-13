import asyncio
import json
import logging
from datetime import datetime
import uuid
from fastapi import APIRouter, HTTPException, Security
from sqlalchemy.exc import DBAPIError, IntegrityError, OperationalError

from app.api.auth import verify_service_key
from app.config import settings
from ray import serve as _ray_serve

def _get_orchestrator():
    return _ray_serve.get_deployment_handle("orchestrator", "default")

logger = logging.getLogger(__name__)

_recon_logger = logging.getLogger("reconstruct")
_recon_logger.setLevel(logging.INFO)
_fh = logging.FileHandler("/workspace/hear-ai/logs/reconstruct.log")
_fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
if not _recon_logger.handlers:
    _recon_logger.addHandler(_fh)
_recon_logger.propagate = False
from app.models.schemas import PipelineRequest, RealtimeRequest, ReconstructRequest, SegmentChange, JobAccepted, EditTranscriptRequest
from app.models.database import SessionLocal, AiJob, AiTrackJob
from app.core.db_gate import commit_with_retry, is_transient_db_error
from app.core.downloader import download_audio
from app.core.hear_temp import cleanup_job_temp, drop_temp_standalone
from app.services.synthesizer import SpeechSynthesizer
from app.services.regeneration.service import RegenerationService

_regeneration_synthesizer = SpeechSynthesizer()
regen_service = RegenerationService(_regeneration_synthesizer)

router = APIRouter(tags=["Pipeline"])
ALLOWED_JOB_TYPES = {"pipeline", "magic-clean", "magic_clean", "rebuild", "reconstruct", "transcription", "categorization", "audio_tag", "edit_transcript"}


def _find_existing_job(db, track_id: str, job_type: str, exclude_job_id: str | None = None) -> str | None:
    existing = db.query(AiJob.id).filter(
        AiJob.track_id == track_id,
        AiJob.job_type == job_type,
        AiJob.status.in_(["queued", "running"]),
        AiJob.id != (exclude_job_id or ""),
    ).first()
    return existing[0] if existing else None


def _submit_job(job_id: str, run_id: str, user_id: str | None = None):
    try:
        _get_orchestrator().process.remote(job_id, run_id)
    except Exception as exc:
        print(f"[PIPELINE] Failed to submit job {job_id}: {exc}")


def _normalize_pipeline_job_type(job_type: str) -> str:
    if job_type == "magic-clean":
        return "magic_clean"
    return job_type


def _track_job_options_payload(body: PipelineRequest | RealtimeRequest, normalized_job_type: str) -> dict | None:
    if normalized_job_type not in ("pipeline", "magic_clean", "audio_tag"):
        return None
    out: dict = {}
    if normalized_job_type == "audio_tag":
        if getattr(body, "type", None):
            out["type"] = body.type
        if getattr(body, "media_file_id", None):
            out["media_file_id"] = body.media_file_id
        if getattr(body, "audio_url", None):
            out["audio_url"] = body.audio_url
    if body.speed_multipliers is not None:
        out["speed_multipliers"] = [float(x) for x in body.speed_multipliers]
    pi = (body.playback_instruction or "").strip() if getattr(body, "playback_instruction", None) else ""
    if pi:
        out["playback_instruction"] = pi
    return out or None


def _resolve_reconstruct_payload(body: PipelineRequest | RealtimeRequest) -> dict:
    changes: list[SegmentChange] = list(body.changes or [])
    if not changes:
        raise HTTPException(
            status_code=422,
            detail="reconstruct submitted via /api/v1/process requires changes[] (array of segments)",
        )
    payload = {
        "changes": [
            {
                "segment_start": c.segment_start,
                "segment_end": c.segment_end,
                "new_text": c.new_text,
                "original_text": c.original_text,
            }
            for c in changes
        ],
        "same_speaker": bool(body.same_speaker),
    }
    if body.audio_url:
        payload["audio_url"] = body.audio_url
    return payload




@router.post(
    "/api/v1/process",
    response_model=JobAccepted,
    status_code=202,
    summary="Submit a track pipeline job",
    description="Enqueues a track-first job for transcription, moderation, and categorization.",
)
async def process_pipeline(body: PipelineRequest, _auth: bool = Security(verify_service_key)):
    normalized_job_type = _normalize_pipeline_job_type(body.job_type)
    if body.job_type not in ALLOWED_JOB_TYPES:
        raise HTTPException(status_code=422, detail=f"Unsupported job_type: {body.job_type}")
    if normalized_job_type == "rebuild" and not (body.edited_transcript or "").strip():
        raise HTTPException(status_code=422, detail="edited_transcript is required for rebuild")
    reconstruct_payload = None
    if normalized_job_type == "reconstruct":
        reconstruct_payload = _resolve_reconstruct_payload(body)
        _recon_logger.info(
            "[RECONSTRUCT-PIPELINE] job_id=%s track_id=%s audio_url=%s same_speaker=%s changes=%s",
            body.job_id, body.track_id, body.audio_url, body.same_speaker,
            json.dumps([
                {"segment_start": c.segment_start, "segment_end": c.segment_end, "new_text": c.new_text, "original_text": c.original_text}
                for c in (body.changes or [])
            ]),
        )
    job_opts = _track_job_options_payload(body, normalized_job_type)
    run_id = str(uuid.uuid4())
    for attempt in range(5):
        db = SessionLocal()
        try:
            if body.track_id:
                dup_id = _find_existing_job(db, body.track_id, normalized_job_type, exclude_job_id=body.job_id)
                if dup_id:
                    return JobAccepted(job_id=dup_id)

            existing = db.query(AiJob).filter(AiJob.id == body.job_id).first()
            if existing:
                existing.run_id = run_id
                existing.status = "queued"
                existing.current_stage = None
                existing.attempts = 0
                existing.started_at = None
                existing.completed_at = None
                existing.error = None
                existing.result_json = None
                existing.job_type = normalized_job_type
                existing.max_tags = body.max_tags
                existing.track_id = body.track_id
                existing.edited_transcript = body.edited_transcript
                existing.input_url = body.audio_url if normalized_job_type in ("reconstruct", "audio_tag") else None
                existing.custom_tags = reconstruct_payload if normalized_job_type == "reconstruct" else None
                existing.job_options = job_opts
                await commit_with_retry(db)
                _submit_job(body.job_id, run_id, body.user_id if hasattr(body, "user_id") else None)
                return JobAccepted(job_id=body.job_id)

            job = AiJob(
                id=body.job_id,
                run_id=run_id,
                job_type=normalized_job_type,
                track_id=body.track_id,
                edited_transcript=body.edited_transcript,
                status="queued",
                
                max_tags=body.max_tags,
                input_url=body.audio_url if normalized_job_type in ("reconstruct", "audio_tag") else None,
                custom_tags=reconstruct_payload if normalized_job_type == "reconstruct" else None,
                job_options=job_opts,
                created_at=datetime.utcnow(),
            )
            db.add(job)
            await commit_with_retry(db)
            _submit_job(body.job_id, run_id, body.user_id if hasattr(body, "user_id") else None)
            return JobAccepted(job_id=body.job_id)
        except IntegrityError:
            db.rollback()
            if body.track_id:
                dup_id = _find_existing_job(db, body.track_id, normalized_job_type, exclude_job_id=body.job_id)
                if dup_id:
                    return JobAccepted(job_id=dup_id)
            _submit_job(body.job_id, run_id, body.user_id if hasattr(body, "user_id") else None)
            return JobAccepted(job_id=body.job_id)
        except (OperationalError, DBAPIError) as exc:
            db.rollback()
            if is_transient_db_error(exc) and attempt < 4:
                await asyncio.sleep(0.15 * (2 ** attempt))
                continue
            raise HTTPException(status_code=503, detail="database_busy_try_again")
        finally:
            db.close()
    raise HTTPException(status_code=503, detail="database_busy_try_again")


@router.post(
    "/api/v1/process-realtime",
    status_code=202,
    summary="Process a track with real-time streaming",
    description="Fetches a track from backend, streams stage progress via SSE/WebSocket, and posts final result to callback.",
)
async def process_realtime(
    body: RealtimeRequest,
    _auth: bool = Security(verify_service_key),
):
    normalized_job_type = _normalize_pipeline_job_type(body.job_type)
    if body.job_type not in ALLOWED_JOB_TYPES:
        raise HTTPException(status_code=422, detail=f"Unsupported job_type: {body.job_type}")
    reconstruct_payload = None
    if normalized_job_type == "reconstruct":
        reconstruct_payload = _resolve_reconstruct_payload(body)
    job_opts = _track_job_options_payload(body, normalized_job_type)
    run_id = str(uuid.uuid4())
    for attempt in range(5):
        db = SessionLocal()
        try:
            if body.track_id:
                dup_id = _find_existing_job(db, body.track_id, normalized_job_type, exclude_job_id=body.job_id)
                if dup_id:
                    return {
                        "job_id": dup_id,
                        "run_id": "",
                        "track_id": body.track_id,
                    }

            existing = db.query(AiJob).filter(AiJob.id == body.job_id).first()
            if existing:
                existing.run_id = run_id
                existing.status = "queued"
                existing.current_stage = None
                existing.attempts = 0
                existing.started_at = None
                existing.completed_at = None
                existing.error = None
                existing.result_json = None
                existing.job_type = normalized_job_type
                existing.max_tags = body.max_tags
                existing.track_id = body.track_id
                existing.input_url = body.audio_url if normalized_job_type in ("reconstruct", "audio_tag") else None
                existing.custom_tags = reconstruct_payload if normalized_job_type == "reconstruct" else None
                existing.job_options = job_opts
                await commit_with_retry(db)
            else:
                job = AiJob(
                    id=body.job_id,
                    run_id=run_id,
                    job_type=normalized_job_type,
                    track_id=body.track_id,
                    status="queued",
                    
                    max_tags=body.max_tags,
                    input_url=body.audio_url if normalized_job_type in ("reconstruct", "audio_tag") else None,
                    custom_tags=reconstruct_payload if normalized_job_type == "reconstruct" else None,
                    job_options=job_opts,
                    created_at=datetime.utcnow(),
                )
                db.add(job)
                await commit_with_retry(db)
            break
        except (OperationalError, DBAPIError) as exc:
            db.rollback()
            if is_transient_db_error(exc) and attempt < 4:
                await asyncio.sleep(0.15 * (2 ** attempt))
                continue
            raise HTTPException(status_code=503, detail="database_busy_try_again")
        finally:
            db.close()
    else:
        raise HTTPException(status_code=503, detail="database_busy_try_again")

    _submit_job(body.job_id, run_id, body.user_id if hasattr(body, "user_id") else None)

    return {
        "job_id": body.job_id,
        "run_id": run_id,
        "track_id": body.track_id,
    }


@router.post(
    "/api/v1/reconstruct",
    summary="Reconstruct an audio segment (preview mode)",
    description="Re-synthesises a segment of track audio with new text. "
    "Returns a preview that the frontend can play. "
    "Send POST /api/v1/reconstruct/confirm with the preview_id to finalize. "
    "Set X-Preview-Mode: false header for legacy direct-splice behavior.",
)
async def reconstruct_segment(body: ReconstructRequest, _auth: bool = Security(verify_service_key)):
    preview_mode = True  # default to new flow
    changes: list[SegmentChange] = list(body.changes or [])
    if not changes:
        if body.segment_start is None or body.segment_end is None or not (body.new_text or "").strip():
            raise HTTPException(
                status_code=422,
                detail="Provide either changes[] or segment_start, segment_end, and new_text",
            )
        changes = [
            SegmentChange(
                segment_start=body.segment_start,
                segment_end=body.segment_end,
                new_text=body.new_text or "",
            )
        ]
    _recon_logger.info(
        "[RECONSTRUCT-DIRECT] track_id=%s audio_url=%s same_speaker=%s changes=%s",
        body.track_id, body.audio_url, body.same_speaker,
        json.dumps([
            {"segment_start": c.segment_start, "segment_end": c.segment_end, "new_text": c.new_text, "original_text": c.original_text}
            for c in changes
        ]),
    )

    if not preview_mode:
        tmp_path = await download_audio(body.audio_url, suffix=".wav")
        try:
            result = await _regeneration_synthesizer.reconstruct_segments(
                    original_audio_path=tmp_path,
                    track_id=body.track_id,
                    changes=[
                        {
                            "segment_start": c.segment_start,
                            "segment_end": c.segment_end,
                            "new_text": c.new_text,
                            "original_text": c.original_text,
                        }
                        for c in changes
                    ],
                    same_speaker=body.same_speaker,
                )
            return {
                "audio_url": result.audio_url,
                "b2_key": result.b2_key,
                "duration": result.duration,
                "segments_applied": len(changes),
            }
        finally:
            drop_temp_standalone(tmp_path)

    changes_dicts = [
        {
            "segment_start": c.segment_start,
            "segment_end": c.segment_end,
            "new_text": c.new_text,
            "original_text": c.original_text,
        }
        for c in changes
    ]

    preview = await regen_service.create_preview(
        track_id=body.track_id,
        audio_url=body.audio_url,
        changes=changes_dicts,
        same_speaker=body.same_speaker,
        user_id=None,
    )

    return {
        "preview_id": preview.preview_id,
        "preview_audio_url": preview.preview_audio_url,
        "preview_duration": preview.preview_duration,
        "quality_metrics": preview.quality_metrics,
        "expires_at": str(preview.expires_at),
        "segments_applied": len(changes),
        "track_id": body.track_id,
    }


@router.post(
    "/api/v1/edit-transcript",
    response_model=JobAccepted,
    status_code=202,
    summary="Edit transcript and reconstruct audio",
    description="Accepts an edited transcript, diffs it against the original, and enqueues a voice-matched audio reconstruction job.",
)
async def edit_transcript(body: EditTranscriptRequest, _auth: bool = Security(verify_service_key)):
    if not (body.edited_transcript or "").strip():
        raise HTTPException(status_code=422, detail="edited_transcript is required")
    _recon_logger.info(
        "[EDIT-TRANSCRIPT] job_id=%s track_id=%s same_speaker=%s edited_transcript=%s",
        body.job_id, body.track_id, body.same_speaker,
        (body.edited_transcript or "")[:200],
    )
    job_opts: dict = {}
    if body.user_id:
        job_opts["user_id"] = body.user_id
    run_id = str(uuid.uuid4())
    for attempt in range(5):
        db = SessionLocal()
        try:
            if body.track_id:
                dup_id = worker.find_existing_job(db, body.track_id, "edit_transcript", exclude_job_id=body.job_id)
                if dup_id:
                    return JobAccepted(job_id=dup_id)

            existing = db.query(AiJob).filter(AiJob.id == body.job_id).first()
            if existing:
                existing.run_id = run_id
                existing.status = "queued"
                existing.current_stage = None
                existing.attempts = 0
                existing.started_at = None
                existing.completed_at = None
                existing.error = None
                existing.result_json = None
                existing.job_type = "edit_transcript"
                existing.track_id = body.track_id
                existing.edited_transcript = body.edited_transcript
                existing.job_options = job_opts or None
                await commit_with_retry(db)
                _submit_job(body.job_id, run_id, body.user_id if hasattr(body, "user_id") else None)
                return JobAccepted(job_id=body.job_id)

            job = AiJob(
                id=body.job_id,
                run_id=run_id,
                job_type="edit_transcript",
                track_id=body.track_id,
                edited_transcript=body.edited_transcript,
                status="queued",
                
                job_options=job_opts or None,
                created_at=datetime.utcnow(),
            )
            db.add(job)
            await commit_with_retry(db)
            _submit_job(body.job_id, run_id, body.user_id if hasattr(body, "user_id") else None)
            return JobAccepted(job_id=body.job_id)
        except IntegrityError:
            db.rollback()
            if body.track_id:
                dup_id = worker.find_existing_job(db, body.track_id, "edit_transcript", exclude_job_id=body.job_id)
                if dup_id:
                    return JobAccepted(job_id=dup_id)
            _submit_job(body.job_id, run_id, body.user_id if hasattr(body, "user_id") else None)
            return JobAccepted(job_id=body.job_id)
        except (OperationalError, DBAPIError) as exc:
            db.rollback()
            if is_transient_db_error(exc) and attempt < 4:
                await asyncio.sleep(0.15 * (2 ** attempt))
                continue
            raise HTTPException(status_code=503, detail="database_busy_try_again")
        finally:
            db.close()
    raise HTTPException(status_code=503, detail="database_busy_try_again")


@router.get(
    "/api/v1/queue/stats",
    tags=["Queue"],
    summary="Get queue stats",
    description="Returns realtime queue position, active count, and estimated wait times.",
)
async def queue_stats(_auth: bool = Security(verify_service_key)):
    try:
        stats = await _get_orchestrator().get_stats.remote()
        return stats
    except Exception:
        return {
            "queued": 0, "active": 0, "total": 0,
            "oldest_wait_s": 0.0, "estimated_wait_s": 0.0,
            "avg_job_duration_s": 30.0,
        }


@router.get(
    "/api/v1/jobs/{job_id}",
    tags=["Jobs"],
    summary="Get job status",
    description="Retrieves the current status and result of a processing job by its ID.",
)
async def get_job(job_id: str, _auth: bool = Security(verify_service_key)):
    db = SessionLocal()
    try:
        job = db.query(AiJob).filter(AiJob.id == job_id).first()
        if not job:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "job_not_found",
                    "job_id": job_id,
                    "message": "No job with this ID exists. It may not have been submitted yet or the ID is incorrect.",
                },
            )
        started = job.started_at
        completed = job.completed_at
        created = job.created_at
        processing_seconds = round((completed - started).total_seconds(), 1) if started and completed else None
        queue_wait_seconds = round((started - created).total_seconds(), 1) if started else round((datetime.utcnow() - created).total_seconds(), 1)
        return {
            "job_id": job.id,
            "run_id": job.run_id,
            "job_type": job.job_type or "pipeline",
            "status": job.status,
            "current_stage": job.current_stage,
            "track_id": job.track_id,
            "attempts": job.attempts,
            "result": job.result_json,
            "error": job.error,
            "created_at": str(created),
            "started_at": str(started) if started else None,
            "completed_at": str(completed) if completed else None,
            "processing_seconds": processing_seconds,
            "queue_wait_seconds": queue_wait_seconds if job.status == "queued" else None,
            "track_state": _get_track_state(db, job.id, job.run_id),
        }
    finally:
        db.close()


@router.post(
    "/api/v1/jobs/{job_id}/cancel",
    tags=["Jobs"],
    status_code=200,
    summary="Cancel a job",
    description="Marks a pending or in-progress job as cancelled. No-op if the job is already completed or failed.",
)
async def cancel_job(job_id: str, _auth: bool = Security(verify_service_key)):
    db = SessionLocal()
    try:
        job = db.query(AiJob).filter(AiJob.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status in ("completed", "failed", "cancelled"):
            return {"job_id": job_id, "status": job.status, "cancelled": False}
        job.status = "cancelled"
        job.current_stage = None
        job.completed_at = datetime.utcnow()
        track = (
            db.query(AiTrackJob)
            .filter(AiTrackJob.job_id == job.id, AiTrackJob.run_id == job.run_id)
            .first()
        )
        if track:
            track.status = "cancelled"
            track.current_stage = None
            track.completed_at = datetime.utcnow()
            track.updated_at = datetime.utcnow()
        await commit_with_retry(db)
        try:
            cleanup_job_temp(db, job.id, job.run_id)
            await commit_with_retry(db)
        except Exception:
            pass
        return {"job_id": job_id, "status": "cancelled", "cancelled": True}
    finally:
        db.close()


def _get_track_state(db, job_id: str, run_id: str | None):
    if not run_id:
        return None
    row = (
        db.query(AiTrackJob)
        .filter(
            AiTrackJob.job_id == job_id,
            AiTrackJob.run_id == run_id,
        )
        .first()
    )
    if not row:
        return None
    return {
        "track_id": row.track_id,
        "status": row.status,
        "current_stage": row.current_stage,
        "attempts": row.attempts,
        "error": row.error,
        "started_at": str(row.started_at) if row.started_at else None,
        "completed_at": str(row.completed_at) if row.completed_at else None,
    }
