import asyncio
from datetime import datetime
import uuid
# todo: check if this is needed
from fastapi import APIRouter, HTTPException, Security, WebSocket, WebSocketDisconnect
from sqlalchemy.exc import IntegrityError

from app.api.auth import verify_service_key
from app.config import settings
from app.models.schemas import PipelineRequest, RealtimeRequest, ReconstructRequest, SegmentChange, JobAccepted
from app.models.database import SessionLocal, AiJob, AiTrackJob
from app.core.downloader import download_audio, cleanup_temp
from app.realtime.broadcaster import manager, make_sse_response
from app.services.registry import worker, synthesizer
from app.services.callback import callback_service

router = APIRouter(tags=["Pipeline"])
ALLOWED_JOB_TYPES = {"pipeline", "magic-clean", "magic_clean", "rebuild", "reconstruct", "transcription", "categorization"}


def _normalize_pipeline_job_type(job_type: str) -> str:
    if job_type == "magic-clean":
        return "magic_clean"
    return job_type


def _resolve_reconstruct_payload(body: PipelineRequest | RealtimeRequest) -> dict:
    changes: list[SegmentChange] = list(body.changes or [])
    if not changes:
        raise HTTPException(
            status_code=422,
            detail="reconstruct submitted via /api/v1/process requires changes[] (array of segments)",
        )
    return {
        "changes": [
            {
                "segment_start": c.segment_start,
                "segment_end": c.segment_end,
                "new_text": c.new_text,
            }
            for c in changes
        ],
        "same_speaker": bool(body.same_speaker),
    }

# todo: check if this is needed
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
    run_id = str(uuid.uuid4())
    db = SessionLocal()
    try:
        existing = db.query(AiJob).filter(AiJob.id == body.job_id).first()
        if existing:
            if existing.status in ("queued", "running"):
                return JobAccepted(job_id=body.job_id)
            existing.run_id = run_id
            existing.status = "queued"
            existing.current_stage = None
            existing.attempts = 0
            existing.started_at = None
            existing.completed_at = None
            existing.error = None
            existing.result_json = None
            existing.callback_delivered = False
            existing.job_type = normalized_job_type
            existing.max_tags = body.max_tags
            existing.track_id = body.track_id
            existing.edited_transcript = body.edited_transcript
            existing.input_url = body.audio_url if normalized_job_type == "reconstruct" else None
            existing.custom_tags = reconstruct_payload if normalized_job_type == "reconstruct" else None
            db.commit()
            worker.enqueue(body.job_id, run_id=run_id)
            return JobAccepted(job_id=body.job_id)

        job = AiJob(
            id=body.job_id,
            run_id=run_id,
            job_type=normalized_job_type,
            track_id=body.track_id,
            edited_transcript=body.edited_transcript,
            status="queued",
            callback_url=settings.HEAR_CALLBACK_URL or None,
            max_tags=body.max_tags,
            input_url=body.audio_url if normalized_job_type == "reconstruct" else None,
            custom_tags=reconstruct_payload if normalized_job_type == "reconstruct" else None,
            created_at=datetime.utcnow(),
        )
        db.add(job)
        db.commit()
    except IntegrityError:
        db.rollback()
        worker.enqueue(body.job_id, run_id=run_id)
        return JobAccepted(job_id=body.job_id)
    finally:
        db.close()

    worker.enqueue(body.job_id, run_id=run_id)
    return JobAccepted(job_id=body.job_id)


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
    run_id = str(uuid.uuid4())
    db = SessionLocal()
    try:
        existing = db.query(AiJob).filter(AiJob.id == body.job_id).first()
        if existing:
            if existing.status in ("queued", "running"):
                return {
                    "job_id": body.job_id,
                    "run_id": existing.run_id,
                    "track_id": existing.track_id,
                    "sse_url": f"/api/v1/events/{body.job_id}",
                    "ws_url": f"/ws/{body.job_id}",
                }
            existing.run_id = run_id
            existing.status = "queued"
            existing.current_stage = None
            existing.attempts = 0
            existing.started_at = None
            existing.completed_at = None
            existing.error = None
            existing.result_json = None
            existing.callback_delivered = False
            existing.job_type = normalized_job_type
            existing.max_tags = body.max_tags
            existing.track_id = body.track_id
            existing.input_url = body.audio_url if normalized_job_type == "reconstruct" else None
            existing.custom_tags = reconstruct_payload if normalized_job_type == "reconstruct" else None
            db.commit()
        else:
            job = AiJob(
                id=body.job_id,
                run_id=run_id,
                job_type=normalized_job_type,
                track_id=body.track_id,
                status="queued",
                callback_url=settings.HEAR_CALLBACK_URL or None,
                max_tags=body.max_tags,
                input_url=body.audio_url if normalized_job_type == "reconstruct" else None,
                custom_tags=reconstruct_payload if normalized_job_type == "reconstruct" else None,
                created_at=datetime.utcnow(),
            )
            db.add(job)
            db.commit()
    finally:
        db.close()

    worker.enqueue(body.job_id, run_id=run_id)

    return {
        "job_id": body.job_id,
        "run_id": run_id,
        "track_id": body.track_id,
        "sse_url": f"/api/v1/events/{body.job_id}",
        "ws_url": f"/ws/{body.job_id}",
    }


@router.post(
    "/api/v1/reconstruct",
    summary="Reconstruct an audio segment",
    description="Re-synthesises a segment of track audio with new text.",
)
async def reconstruct_segment(body: ReconstructRequest, _auth: bool = Security(verify_service_key)):
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
    tmp_path = await download_audio(body.audio_url)
    try:
        result = await synthesizer.reconstruct_segments(
            original_audio_path=tmp_path,
            track_id=body.track_id,
            changes=changes,
            same_speaker=body.same_speaker,
        )
        return {
            "audio_url": result.audio_url,
            "b2_key": result.b2_key,
            "duration": result.duration,
            "segments_applied": len(changes),
        }
    finally:
        cleanup_temp(tmp_path)


@router.get(
    "/api/v1/events/{job_id}",
    tags=["Realtime"],
    summary="Subscribe to job events (SSE)",
    description="Opens a Server-Sent Events stream for real-time pipeline progress updates.",
)
async def sse_stream(job_id: str, _auth: bool = Security(verify_service_key)):
    return make_sse_response(job_id)


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
            "callback_delivered": job.callback_delivered,
            "created_at": str(job.created_at),
            "started_at": str(job.started_at) if job.started_at else None,
            "completed_at": str(job.completed_at) if job.completed_at else None,
            "track_state": _get_track_state(job.id, job.run_id),
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
        db.commit()
        return {"job_id": job_id, "status": "cancelled", "cancelled": True}
    finally:
        db.close()


@router.post(
    "/api/v1/jobs/{job_id}/retry-callback",
    tags=["Jobs"],
    summary="Retry callback delivery",
    description="Re-sends the job result to the callback URL. Use when the backend missed the original delivery.",
)
async def retry_callback(job_id: str, _auth: bool = Security(verify_service_key)):
    db = SessionLocal()
    try:
        job = db.query(AiJob).filter(AiJob.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status not in ("completed", "failed"):
            raise HTTPException(status_code=409, detail="Job still processing")

        effective_url = job.callback_url or settings.HEAR_CALLBACK_URL
        if not effective_url:
            raise HTTPException(status_code=400, detail="No callback URL configured")

        if job.status == "completed":
            payload = {
                "job_id": job.id,
                "run_id": job.run_id,
                "track_id": job.track_id,
                "job_type": job.job_type or "pipeline",
                "status": "completed",
                "result": job.result_json or {},
                "error": None,
            }
        else:
            payload = {
                "job_id": job.id,
                "run_id": job.run_id,
                "track_id": job.track_id,
                "job_type": job.job_type or "pipeline",
                "status": "failed",
                "result": None,
                "error": job.error or "unknown",
            }

        delivered = await callback_service.send(effective_url, payload)
        job.callback_delivered = delivered
        db.commit()

        if delivered:
            return {"status": "delivered", "job_id": job.id}
        raise HTTPException(status_code=502, detail="Callback delivery failed")
    finally:
        db.close()


@router.websocket("/ws/{job_id}")
async def websocket_stream(ws: WebSocket, job_id: str):
    await manager.connect_ws(job_id, ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect_ws(job_id, ws)


def _get_track_state(job_id: str, run_id: str | None):
    if not run_id:
        return None
    db = SessionLocal()
    try:
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
    finally:
        db.close()
