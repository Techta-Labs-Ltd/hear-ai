from fastapi import APIRouter, Security

from app.api.auth import verify_service_key
from app.core.category_loader import category_loader
from app.core.discovery_taxonomy import discovery_taxonomy_loader
from app.models.schemas import CategorizeRequest, CategorizeResponse, TaxonomySyncResponse
from app.services.categorizer import CategorizationService

_router_categorizer = CategorizationService()

router = APIRouter(prefix="/api/v1", tags=["Categorization"])


@router.post(
    "/categorize",
    response_model=CategorizeResponse,
    summary="Categorize text content",
    description=(
        "Analyzes transcript text to assign topic tags and categories. "
        "When Qwen is enabled (QWEN_LLM_ENABLED), the LLM is the primary classifier and reads the "
        "full transcript plus discovery taxonomy context; keyword/NLI layers only advise and lightly validate. "
        "Any new tags or categories are persisted to data/categories.txt via category_loader.ensure_labels. "
        "Tags are always #hashtags; discovery taxonomy paths (with ' > ') are not returned as tags. "
        "Response includes new_tags_added, new_categories_added, llm_used, and categorizer_mode."
    ),
)
async def categorize(body: CategorizeRequest, _auth: bool = Security(verify_service_key)):
    result = await _router_categorizer.categorize(
        transcript=body.text,
        custom_tags=body.custom_tags,
        max_tags=body.max_tags,
    )
    return result


@router.post(
    "/admin/sync-taxonomy",
    response_model=TaxonomySyncResponse,
    summary="Sync discovery taxonomy into categories catalog",
    description=(
        "Merges paths from data/discovery_taxonomy.txt into data/categories.txt "
        "(categories, slug tags, and persistence). Same logic as startup when "
        "CATEGORIZER_SYNC_TAXONOMY is enabled. Safe to call repeatedly — only adds missing labels."
    ),
)
async def sync_taxonomy(_auth: bool = Security(verify_service_key)):
    discovery_taxonomy_loader.load()
    category_loader.load()
    tags_added, cats_added = category_loader.import_discovery_taxonomy(
        discovery_taxonomy_loader.data.paths
    )
    data = category_loader.data
    return TaxonomySyncResponse(
        tags_added=tags_added,
        categories_added=cats_added,
        total_tags=len(data.tags),
        total_categories=len(data.categories),
    )
