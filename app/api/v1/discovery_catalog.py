from fastapi import APIRouter, Query, Security

from app.api.auth import verify_service_key
from app.core.discovery_sort import VALID_DISCOVERY_SORTS, sort_discovery_items
from app.models.database import SessionLocal, AiTrackJob
from app.models.schemas import DiscoveryCatalogItem, DiscoveryCatalogResponse

router = APIRouter(prefix="/api/v1", tags=["Discovery"])


@router.get(
    "/discovery/catalog",
    response_model=DiscoveryCatalogResponse,
    summary="List discovery profiles for browse feeds",
    description=(
        "Returns completed track discovery metadata stored by Hear AI. "
        "Use sort=latest (default) for recency by latest_at/published_at, "
        "or sort=trending for trending_score descending then recency."
    ),
)
async def list_discovery_catalog(
    sort: str = Query(
        "latest",
        description="Sort order: latest (recency) or trending (score then recency)",
    ),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    _auth: bool = Security(verify_service_key),
):
    mode = (sort or "latest").strip().lower()
    if mode not in VALID_DISCOVERY_SORTS:
        mode = "latest"

    db = SessionLocal()
    try:
        query = db.query(AiTrackJob).filter(
            AiTrackJob.status == "completed",
            AiTrackJob.discovery_json.isnot(None),
        )
        if mode == "latest":
            query = query.order_by(AiTrackJob.completed_at.desc())
        items: list[dict] = []
        for row in query.all():
            discovery = row.discovery_json
            if not isinstance(discovery, dict) or not discovery:
                continue
            item = dict(discovery)
            item.setdefault("content_id", row.track_id)
            item.setdefault("latest_at", item.get("published_at") or item.get("created_at"))
            item.setdefault("published_at", item.get("published_at") or "")
            item.setdefault("trending_score", item.get("trending_score", 0))
            items.append(
                {
                    "track_id": row.track_id,
                    "job_id": row.job_id,
                    "discovery": item,
                    "latest_at": str(item.get("latest_at") or ""),
                    "published_at": str(item.get("published_at") or ""),
                    "trending_score": float(item.get("trending_score") or 0),
                    "completed_at": (
                        row.completed_at.isoformat() if row.completed_at else None
                    ),
                }
            )

        sorted_items = sort_discovery_items(items, mode)
        total = len(sorted_items)
        page = sorted_items[offset : offset + limit]
        return DiscoveryCatalogResponse(
            sort=mode,
            limit=limit,
            offset=offset,
            total=total,
            items=[DiscoveryCatalogItem.model_validate(row) for row in page],
        )
    finally:
        db.close()
