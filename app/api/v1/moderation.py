from fastapi import APIRouter, Depends

from app.api.auth import verify_service_key
from app.models.schemas import ModerateRequest
from app.services.registry import moderator

router = APIRouter(prefix="/api/v1", tags=["Moderation"])


@router.post("/moderate")
async def moderate(body: ModerateRequest, _auth: bool = Depends(verify_service_key)):
    result = await moderator.moderate(body.text)
    return result
