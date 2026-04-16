from fastapi import APIRouter, Security
from pydantic import BaseModel

from app.api.auth import verify_service_key
from app.models.schemas import CategorizeRequest
from app.services.registry import categorizer

router = APIRouter(prefix="/api/v1", tags=["Categorization"])


class CategorizeResponse(BaseModel):
    tags: list[str]
    categories: list[str]
    confidence_scores: dict[str, float]
    sentiment: str
    new_tags_added: list[str]
    new_categories_added: list[str]
    settings_applied: bool


@router.post(
    "/categorize",
    response_model=CategorizeResponse,
    summary="Categorize text content",
    description=(
        "Analyzes transcript text to assign relevant topic tags and categories. "
        "Uses three layers: keyword rules, zero-shot NLI classification, and an optional OpenAI LLM pass. "
        "Platform settings (auto_tag_keywords, blocked_keywords) are fetched and applied automatically. "
        "Any new tags or categories discovered from the transcript are persisted to data/categories.txt. "
        "Response includes `new_tags_added` and `new_categories_added` lists so callers can track data growth."
    ),
)
async def categorize(body: CategorizeRequest, _auth: bool = Security(verify_service_key)):
    result = await categorizer.categorize(
        transcript=body.text,
        custom_tags=body.custom_tags,
        max_tags=body.max_tags,
    )
    return result
