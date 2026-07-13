import asyncio
import logging

from fastapi import APIRouter, HTTPException, Security
from sqlalchemy.exc import DBAPIError, OperationalError

from app.api.auth import verify_service_key
from app.config import settings
from app.core.db_gate import commit_with_retry, is_transient_db_error
from app.core.downloader import download_audio
from app.core.hear_temp import drop_temp_standalone
from app.models.database import SessionLocal, RegenerationPreview
from app.models.schemas import ReconstructConfirmRequest, ReconstructRemoveRequest, JobAccepted
from app.services.synthesizer import SpeechSynthesizer
from app.services.regeneration.service import RegenerationService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/reconstruct", tags=["Reconstruct"])

regen_service = RegenerationService(SpeechSynthesizer())


@router.post(
    "/confirm",
    summary="Confirm regeneration preview",
    description="Takes a preview_id from a previous /api/v1/reconstruct call, "
    "performs the final splice at the exact timestamps, and returns the final audio URL.",
)
async def confirm_reconstruction(
    body: ReconstructConfirmRequest,
    _auth: bool = Security(verify_service_key),
):
    try:
        result = await regen_service.confirm_preview(body.preview_id)
        return {
            "audio_url": result.audio_url,
            "b2_key": result.b2_key,
            "duration": result.duration,
            "track_id": body.track_id,
            "user_id": body.user_id,
            "job_type": "reconstruct",
            "action": "confirm",
            "status": "completed",
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Confirm reconstruction failed: %s", e)
        raise HTTPException(status_code=500, detail="Processing error. Please try again later.")


@router.post(
    "/remove",
    summary="Remove a segment from audio",
    description="Removes a segment from the original audio at the given timestamps. "
    "Returns the full audio file after the cut immediately (no preview needed).",
)
async def remove_segment(
    body: ReconstructRemoveRequest,
    _auth: bool = Security(verify_service_key),
):
    try:
        result = await regen_service.remove_segment(
            track_id=body.track_id,
            audio_url=body.audio_url,
            segment_start=body.segment_start,
            segment_end=body.segment_end,
            user_id=body.user_id,
        )
        return {
            "audio_url": result.audio_url,
            "b2_key": result.b2_key,
            "duration": result.duration,
            "segments_removed": 1,
            "removed_duration": round(body.segment_end - body.segment_start, 3),
            "track_id": body.track_id,
            "user_id": body.user_id,
            "job_type": "reconstruct",
            "action": "remove",
            "status": "completed",
        }
    except Exception as e:
        logger.error("Remove segment failed: %s", e)
        raise HTTPException(status_code=500, detail="Processing error. Please try again later.")


@router.post(
    "/rollback",
    summary="Rollback a regeneration preview",
    description="Marks a pending preview as rolled_back and deletes its B2 assets.",
)
async def rollback_preview(
    preview_id: str,
    _auth: bool = Security(verify_service_key),
):
    success = await regen_service.rollback_preview(preview_id)
    if not success:
        raise HTTPException(status_code=404, detail="Preview not found or already rolled back")
    return {"preview_id": preview_id, "status": "rolled_back"}


@router.get(
    "/previews/{preview_id}",
    summary="Get preview status and metrics",
    description="Returns the current status and quality metrics for a regeneration preview.",
)
async def get_preview(
    preview_id: str,
    _auth: bool = Security(verify_service_key),
):
    preview = await regen_service.get_preview(preview_id)
    if not preview:
        raise HTTPException(status_code=404, detail="Preview not found")
    return preview
