from fastapi import APIRouter, Security

from app.api.auth import verify_service_key
from app.models.schemas import ModerateRequest
from app.services.moderator import ModerationService

_router_moderator = ModerationService()

router = APIRouter(prefix="/api/v1", tags=["Moderation"])


@router.post(
    "/moderate",
    summary="Moderate text content",
    description="Analyzes text for content safety, returning a moderation verdict with flagged categories and confidence scores.",
)
async def moderate(body: ModerateRequest, _auth: bool = Security(verify_service_key)):
    result = await _router_moderator.moderate(body.text)
    return result
