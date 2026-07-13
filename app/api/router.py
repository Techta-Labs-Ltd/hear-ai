from fastapi import APIRouter

from app.api.v1.health import router as health_router
from app.api.v1.transcription import router as transcription_router
from app.api.v1.enhancement import router as enhancement_router
from app.api.v1.categorization import router as categorization_router
from app.api.v1.discovery_catalog import router as discovery_catalog_router
from app.api.v1.moderation import router as moderation_router
from app.api.v1.pipeline import router as pipeline_router
from app.api.v1.reconstruct_actions import router as reconstruct_actions_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(transcription_router)
api_router.include_router(enhancement_router)
api_router.include_router(categorization_router)
api_router.include_router(discovery_catalog_router)
api_router.include_router(moderation_router)
api_router.include_router(pipeline_router)
api_router.include_router(reconstruct_actions_router)
