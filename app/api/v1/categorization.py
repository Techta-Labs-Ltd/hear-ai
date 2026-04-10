from fastapi import APIRouter, Depends

from app.api.auth import verify_service_key
from app.models.schemas import CategorizeRequest
from app.services.registry import categorizer

router = APIRouter(prefix="/api/v1", tags=["Categorization"])


@router.post(
    "/categorize",
    summary="Categorize text content",
    description="Analyzes transcript text to assign relevant topic tags using the LLM categorizer. Supports custom tag dictionaries.",
)
async def categorize(body: CategorizeRequest, _auth: bool = Depends(verify_service_key)):
    result = await categorizer.categorize(
        transcript=body.text,
        custom_tags=body.custom_tags,
        max_tags=body.max_tags,
    )
    return result
