from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy.exc import IntegrityError

from app.api.auth import verify_service_key
from app.config import settings
from app.models.schemas import TranscribeRequest, JobAccepted
from app.models.database import SessionLocal, AiJob
from app.services.registry import worker

router = APIRouter(prefix="/api/v1", tags=["Transcription"])


@router.post(
    "/transcribe",
    response_model=JobAccepted,
    status_code=202,
    summary="Submit a transcription job",
    description="Enqueues a recording for standalone speech-to-text transcription using Faster-Whisper.",
)
async def transcribe(body: TranscribeRequest, _auth: bool = Depends(verify_service_key)):
    db = SessionLocal()
    try:
        existing = db.query(AiJob).filter(AiJob.id == body.job_id).first()
        if existing:
            if existing.status not in ("completed", "failed", "cancelled"):
                return JobAccepted(job_id=body.job_id)
            existing.status = "pending"
            existing.attempts = 0
            existing.error = None
            existing.result_json = None
            existing.callback_delivered = False
            db.commit()
            worker.enqueue(body.job_id)
            return JobAccepted(job_id=body.job_id)

        job = AiJob(
            id=body.job_id,
            job_type="transcription",
            recording_id=body.recording_id,
            status="pending",
            callback_url=settings.HEAR_CALLBACK_URL or None,
            skip_enhancement=True,
            created_at=datetime.utcnow(),
        )
        db.add(job)
        db.commit()
    except IntegrityError:
        db.rollback()
        worker.enqueue(body.job_id)
        return JobAccepted(job_id=body.job_id)
    finally:
        db.close()

    worker.enqueue(body.job_id)
    return JobAccepted(job_id=body.job_id)
