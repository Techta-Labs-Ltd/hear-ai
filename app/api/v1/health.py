import torch
from fastapi import APIRouter

from app.api.auth import verify_service_key
from app.core.gpu import gpu
from app.models.schemas import HealthResponse

router = APIRouter(tags=["System"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description="Returns the current health status including GPU availability, loaded ML models, and job queue depth.",
)
async def health() -> HealthResponse:
    gpu_name = ""
    gpu_available = torch.cuda.is_available()
    gpu_memory: dict[str, float] = {}
    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        gpu_memory = {
            "free_mb": round(free / 1e6, 1),
            "used_mb": round((total - free) / 1e6, 1),
            "total_mb": round(total / 1e6, 1),
        }

    return HealthResponse(
        status="healthy",
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        models_loaded=[],
        active_jobs=0,
        queued_jobs=0,
        gpu_memory=gpu_memory,
        redis_status="disabled",
    )
