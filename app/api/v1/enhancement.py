from datetime import datetime

from fastapi import APIRouter, Depends

from app.api.auth import verify_service_key
from app.models.schemas import EnhanceRequest, JobAccepted
from app.models.database import SessionLocal, AiJob
from app.services.registry import worker

router = APIRouter(prefix="/api/v1", tags=["Enhancement"])


@router.post(
    "/enhance",
    response_model=JobAccepted,
    status_code=202,
    summary="Submit an enhancement job",
    description="Enqueues an audio file for standalone vocal isolation and noise removal using Demucs. Transcription is skipped.",
)
async def enhance(body: EnhanceRequest, _auth: bool = Depends(verify_service_key)):
    db = SessionLocal()
    try:
        job = AiJob(
            id=body.job_id,
            job_type="enhancement",
            recording_id=body.recording_id,
            status="pending",
            input_url=body.audio_url,
            callback_url=body.callback_url,
            skip_transcription=True,
            created_at=datetime.utcnow(),
        )
        db.add(job)
        db.commit()
    finally:
        db.close()

    worker.enqueue(body.job_id)

    return JobAccepted(job_id=body.job_id)
