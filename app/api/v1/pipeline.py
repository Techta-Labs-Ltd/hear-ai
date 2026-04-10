import tempfile
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from app.api.auth import verify_service_key
from app.models.schemas import PipelineRequest, ReconstructRequest, JobAccepted
from app.models.database import SessionLocal, AiJob
from app.core.downloader import download_audio, cleanup_temp
from app.realtime.broadcaster import manager, make_sse_response
from app.services.registry import worker, orchestrator, synthesizer
from app.services.callback import callback_service

router = APIRouter(tags=["Pipeline"])


@router.post(
    "/api/v1/process",
    response_model=JobAccepted,
    status_code=202,
    summary="Submit a full pipeline job",
    description="Enqueues an audio file for the full pipeline: enhancement → transcription → categorization → moderation. Returns immediately with a job ID.",
)
async def process_pipeline(body: PipelineRequest, _auth: bool = Depends(verify_service_key)):
    db = SessionLocal()
    try:
        job = AiJob(
            id=body.job_id,
            job_type="pipeline",
            recording_id=body.recording_id,
            status="pending",
            callback_url=body.callback_url,
            skip_enhancement=body.skip_enhancement,
            skip_transcription=body.skip_transcription,
            existing_transcript=body.existing_transcript,
            created_at=datetime.utcnow(),
        )
        db.add(job)
        db.commit()
    finally:
        db.close()

    worker.enqueue(body.job_id)

    return JobAccepted(job_id=body.job_id)


@router.post(
    "/api/v1/process-realtime",
    status_code=202,
    summary="Process audio with real-time streaming",
    description="Uploads an audio file and processes it in the background, streaming progress via SSE and WebSocket.",
)
async def process_realtime(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    recording_id: Optional[str] = None,
    _auth: bool = Depends(verify_service_key),
):
    job_id = str(uuid.uuid4())
    rec_id = recording_id or job_id
    audio_bytes = await file.read()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    background_tasks.add_task(
        orchestrator.process_and_stream,
        job_id=job_id,
        recording_id=rec_id,
        audio_bytes=audio_bytes,
        input_path=tmp_path,
    )

    return {
        "job_id": job_id,
        "recording_id": rec_id,
        "sse_url": f"/api/v1/events/{job_id}",
        "ws_url": f"/ws/{job_id}",
    }


@router.post(
    "/api/v1/reconstruct",
    summary="Reconstruct an audio segment",
    description="Re-synthesises a segment of audio with new text using accent-aware Edge-TTS, returning the replacement audio URL.",
)
async def reconstruct_segment(body: ReconstructRequest, _auth: bool = Depends(verify_service_key)):
    tmp_path = await download_audio(body.audio_url)
    try:
        result = await synthesizer.reconstruct_segment(
            original_audio_path=tmp_path,
            segment_start=body.segment_start,
            segment_end=body.segment_end,
            new_text=body.new_text,
            recording_id=body.recording_id,
        )
        return {
            "audio_url": result.audio_url,
            "b2_key": result.b2_key,
            "duration": result.duration,
        }
    finally:
        cleanup_temp(tmp_path)


@router.get(
    "/api/v1/events/{job_id}",
    tags=["Realtime"],
    summary="Subscribe to job events (SSE)",
    description="Opens a Server-Sent Events stream for real-time pipeline progress updates.",
)
async def sse_stream(job_id: str, _auth: bool = Depends(verify_service_key)):
    return make_sse_response(job_id)


@router.get(
    "/api/v1/jobs/{job_id}",
    tags=["Jobs"],
    summary="Get job status",
    description="Retrieves the current status and result of a processing job by its ID.",
)
async def get_job(job_id: str, _auth: bool = Depends(verify_service_key)):
    db = SessionLocal()
    try:
        job = db.query(AiJob).filter(AiJob.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return {
            "job_id": job.id,
            "status": job.status,
            "recording_id": job.recording_id,
            "result": job.result_json,
            "error": job.error,
            "callback_delivered": job.callback_delivered,
            "created_at": str(job.created_at),
            "completed_at": str(job.completed_at) if job.completed_at else None,
        }
    finally:
        db.close()


@router.post(
    "/api/v1/jobs/{job_id}/retry-callback",
    tags=["Jobs"],
    summary="Retry callback delivery",
    description="Re-sends the job result to the callback URL. Use when the backend missed the original delivery.",
)
async def retry_callback(job_id: str, _auth: bool = Depends(verify_service_key)):
    db = SessionLocal()
    try:
        job = db.query(AiJob).filter(AiJob.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status not in ("completed", "failed"):
            raise HTTPException(status_code=409, detail="Job still processing")
        if not job.callback_url:
            raise HTTPException(status_code=400, detail="No callback URL configured")

        if job.status == "completed":
            payload = {
                "job_id": job.id,
                "job_type": job.job_type or "pipeline",
                "status": "completed",
                "result": job.result_json or {},
            }
        else:
            payload = {
                "job_id": job.id,
                "job_type": job.job_type or "pipeline",
                "status": "failed",
                "error": job.error or "unknown",
            }

        delivered = await callback_service.send(job.callback_url, payload)
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

