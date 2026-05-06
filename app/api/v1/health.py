from fastapi import APIRouter
import importlib.util

from app.api.auth import verify_service_key
from app.config import settings
from app.core.gpu import gpu
from app.models.schemas import HealthResponse
from app.services.registry import transcriber, enhancer, categorizer, moderator

router = APIRouter(tags=["System"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description="Returns the current health status including GPU availability, loaded ML models, and job queue depth.",
)
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
    module_name = (settings.HIGGS_AUDIO_MODULE or "higgs_audio").strip()
    if importlib.util.find_spec(module_name) is not None:
        models_loaded.append(module_name)

    return HealthResponse(
        status="healthy",
        gpu_available=gpu.is_available,
        gpu_name=gpu.gpu_name,
        models_loaded=models_loaded,
        active_jobs=gpu.active_jobs,
        queued_jobs=gpu.queued_jobs,
    )
