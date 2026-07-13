from datetime import datetime
import asyncio
import uuid

from fastapi import APIRouter, HTTPException, Security
from sqlalchemy.exc import DBAPIError, IntegrityError, OperationalError

from app.api.auth import verify_service_key
from app.config import settings
from app.core.db_gate import commit_with_retry, is_transient_db_error
from app.models.schemas import TranscribeRequest, JobAccepted
from app.models.database import SessionLocal, AiJob
from ray import serve as _ray_serve

def _get_orchestrator():
    return _ray_serve.get_deployment_handle("orchestrator", "default")

router = APIRouter(prefix="/api/v1", tags=["Transcription"])

def _find_dup(db, track_id: str, exclude_job_id: str | None = None) -> str | None:
    existing = db.query(AiJob.id).filter(
        AiJob.track_id == track_id,
        AiJob.job_type == "transcription",
        AiJob.status.in_(["queued", "running"]),
        AiJob.id != (exclude_job_id or ""),
    ).first()
    return existing[0] if existing else None

def _submit(job_id: str, run_id: str, user_id: str | None = None):
    try:
        ref = _get_orchestrator().process.remote(job_id, run_id)
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
        loop.create_task(asyncio.ensure_future(asyncio.sleep(0)))
    except Exception as exc:
        print(f"[TRANS] _submit FAIL: {exc}")


@router.post(
    "/transcribe",
    response_model=JobAccepted,
    status_code=202,
    summary="Submit a transcription job",
    description="Enqueues a track for standalone speech-to-text transcription using Faster-Whisper.",
)
async def transcribe(body: TranscribeRequest, _auth: bool = Security(verify_service_key)):
    run_id = str(uuid.uuid4())
    user_id = body.user_id if hasattr(body, "user_id") else None
    for attempt in range(5):
        db = SessionLocal()
        try:
            if body.track_id:
                dup_id = _find_dup(db, body.track_id, exclude_job_id=body.job_id)
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
                existing.job_type = "transcription"
                existing.track_id = body.track_id
                await commit_with_retry(db)
                _submit(body.job_id, run_id, user_id)
                return JobAccepted(job_id=body.job_id)

            job = AiJob(
                id=body.job_id,
                run_id=run_id,
                job_type="transcription",
                track_id=body.track_id,
                status="queued",
                
                skip_enhancement=True,
                created_at=datetime.utcnow(),
            )
            db.add(job)
            await commit_with_retry(db)
            _submit(body.job_id, run_id, user_id)
            return JobAccepted(job_id=body.job_id)
        except IntegrityError:
            db.rollback()
            if body.track_id:
                dup_id = _find_dup(db, body.track_id, exclude_job_id=body.job_id)
                if dup_id:
                    return JobAccepted(job_id=dup_id)
            _submit(body.job_id, run_id, user_id)
            return JobAccepted(job_id=body.job_id)
        except (OperationalError, DBAPIError) as exc:
            db.rollback()
            if is_transient_db_error(exc) and attempt < 4:
                await asyncio.sleep(0.15 * (2 ** attempt))
                continue
            raise
        finally:
            db.close()
    raise HTTPException(status_code=503, detail="database_busy_try_again")
