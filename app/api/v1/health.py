from fastapi import APIRouter, Depends

from app.api.auth import verify_service_key
from app.core.gpu import gpu
from app.models.schemas import HealthResponse
from app.services.registry import transcriber, enhancer, categorizer, moderator

router = APIRouter(tags=["System"])


@router.get("/health", response_model=HealthResponse)
async def health():
    models_loaded = []
    if transcriber.is_loaded:
        models_loaded.append("whisper")
    if enhancer.is_loaded:
        models_loaded.append("demucs")
    if categorizer.is_loaded:
        models_loaded.append("categorizer")
    if moderator.is_loaded:
        models_loaded.append("moderator")

    return HealthResponse(
        status="healthy",
        gpu_available=gpu.is_available,
        gpu_name=gpu.gpu_name,
        models_loaded=models_loaded,
        active_jobs=gpu.active_jobs,
        queued_jobs=gpu.queued_jobs,
    )
