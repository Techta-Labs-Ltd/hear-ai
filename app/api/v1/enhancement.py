from datetime import datetime
import asyncio
import uuid

from fastapi import APIRouter, HTTPException, Security
from sqlalchemy.exc import IntegrityError, OperationalError

from app.api.auth import verify_service_key
from app.config import settings
from app.core.db_gate import db_write_lock
from app.models.schemas import EnhanceRequest, JobAccepted
from app.models.database import SessionLocal, AiJob
from app.services.registry import worker

router = APIRouter(prefix="/api/v1", tags=["Enhancement"])


def _is_locked_error(exc: Exception) -> bool:
    return "database is locked" in str(exc).lower()


async def _commit_with_retry(db, retries: int = 5):
    for attempt in range(retries):
        try:
            async with db_write_lock:
                db.commit()
            return
        except OperationalError as exc:
            db.rollback()
            if _is_locked_error(exc) and attempt < retries - 1:
                await asyncio.sleep(0.15 * (2 ** attempt))
                continue
            raise


@router.post(
    "/enhance",
    response_model=JobAccepted,
    status_code=202,
    summary="Submit an enhancement job",
    description="Enqueues a track for standalone vocal isolation and noise removal using Demucs.",
)
async def enhance(body: EnhanceRequest, _auth: bool = Security(verify_service_key)):
    run_id = str(uuid.uuid4())
    for attempt in range(5):
        db = SessionLocal()
        try:
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
                existing.callback_delivered = False
                existing.job_type = "magic_clean"
                existing.track_id = body.track_id
                await _commit_with_retry(db)
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
            await _commit_with_retry(db)
            worker.enqueue(body.job_id, run_id=run_id)
            return JobAccepted(job_id=body.job_id)
        except IntegrityError:
            db.rollback()
            worker.enqueue(body.job_id, run_id=run_id)
            return JobAccepted(job_id=body.job_id)
        except OperationalError as exc:
            db.rollback()
            if _is_locked_error(exc) and attempt < 4:
                await asyncio.sleep(0.15 * (2 ** attempt))
                continue
            raise
        finally:
            db.close()
    raise HTTPException(status_code=503, detail="database_busy_try_again")
