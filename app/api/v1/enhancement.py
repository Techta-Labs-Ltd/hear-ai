from datetime import datetime
import uuid

from fastapi import APIRouter, Security
from sqlalchemy.exc import IntegrityError

from app.api.auth import verify_service_key
from app.config import settings
from app.models.schemas import EnhanceRequest, JobAccepted
from app.models.database import SessionLocal, AiJob
from app.services.registry import worker

router = APIRouter(prefix="/api/v1", tags=["Enhancement"])


@router.post(
    "/enhance",
    response_model=JobAccepted,
    status_code=202,
    summary="Submit an enhancement job",
    description="Enqueues a track for standalone vocal isolation and noise removal using Demucs.",
)
async def enhance(body: EnhanceRequest, _auth: bool = Security(verify_service_key)):
    run_id = str(uuid.uuid4())
    db = SessionLocal()
    try:
        existing = db.query(AiJob).filter(AiJob.id == body.job_id).first()
        if existing:
            if existing.status in ("queued", "running") and existing.job_type == "magic_clean":
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
            existing.job_type = "magic_clean"
            existing.track_id = body.track_id
            db.commit()
            worker.enqueue(body.job_id, run_id=run_id)
            return JobAccepted(job_id=body.job_id)

        job = AiJob(
            id=body.job_id,
            run_id=run_id,
            job_type="magic_clean",
            track_id=body.track_id,
            status="queued",
            callback_url=settings.HEAR_CALLBACK_URL or None,
            skip_transcription=True,
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
