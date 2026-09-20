from __future__ import annotations

import uuid

from hear.core.backend_registry import validate_storage_for_backend
from hear.core.category_loader import category_loader
from hear.core.discovery_sort import VALID_DISCOVERY_SORTS, sort_discovery_items
from hear.core.health import ServiceHealth
from hear.core.keyword_loader import auto_tag_keyword_loader, harm_keyword_loader
from hear.core.storage import B2Storage
from hear.models.database import AiTrackJob, SessionLocal
from hear.models.schemas import StorageContext
from hear.services.categorization.service import CategorizationService
from hear.services.moderation.service import ModerationService
from hear.services.reconstruction.service import RegenerationService


class ServiceError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class Operations:
    def __init__(
        self,
        moderator: ModerationService,
        categorizer: CategorizationService,
        regeneration: RegenerationService,
        health: ServiceHealth,
    ) -> None:
        self._moderator = moderator
        self._categorizer = categorizer
        self._regeneration = regeneration
        self._health = health

    async def moderate(self, text: str) -> dict:
        return await self._moderator.moderate(text)

    async def categorize(self, text: str, custom_tags: list[str], max_tags: int) -> dict:
        return await self._categorizer.categorize(
            transcript=text,
            custom_tags=custom_tags,
            max_tags=max_tags,
        )

    async def create_preview(
        self,
        *,
        audio_url: str,
        track_id: str,
        changes: list[dict],
        segment_start: float | None,
        segment_end: float | None,
        new_text: str | None,
        same_speaker: bool,
        backend_id: str,
        storage_context: StorageContext,
    ) -> dict:
        validate_storage_for_backend(backend_id, storage_context)
        storage = B2Storage(storage_context)
        if not changes:
            if segment_start is None or segment_end is None or not (new_text or "").strip():
                raise ServiceError(
                    422,
                    "provide changes or segment_start, segment_end, and new_text",
                )
            changes = [
                {
                    "segment_start": segment_start,
                    "segment_end": segment_end,
                    "new_text": new_text,
                    "original_text": None,
                }
            ]
        preview = await self._regeneration.create_preview(
            track_id=track_id,
            audio_url=audio_url,
            changes=changes,
            same_speaker=same_speaker,
            backend_id=backend_id,
            storage=storage,
        )
        return {
            "preview_id": preview.preview_id,
            "preview_audio_url": preview.preview_audio_url,
            "b2_key": preview.preview_b2_key,
            "bucket_name": storage.bucket_name,
            "backend_id": backend_id,
            "preview_duration": preview.preview_duration,
            "quality_metrics": preview.quality_metrics,
            "expires_at": str(preview.expires_at),
            "segments_applied": len(changes),
            "track_id": track_id,
            "segments": [
                {
                    "segment_start": segment.segment_start,
                    "segment_end": segment.segment_end,
                    "b2_key": segment.b2_key,
                    "audio_url": segment.audio_url,
                    "duration": segment.duration,
                    "is_deletion": segment.is_deletion,
                }
                for segment in preview.segments
            ],
        }

    async def confirm_preview(
        self,
        preview_id: str,
        track_id: str | None,
        user_id: str | None,
        backend_id: str,
    ) -> dict:
        try:
            result = await self._regeneration.confirm_preview(preview_id, backend_id)
        except ValueError as exc:
            raise ServiceError(404, str(exc)) from exc
        return {
            "audio_url": result.audio_url,
            "b2_key": result.b2_key,
            "duration": result.duration,
            "bucket_name": result.bucket_name,
            "backend_id": backend_id,
            "track_id": track_id,
            "user_id": user_id,
            "job_type": "reconstruct",
            "action": "confirm",
            "status": "completed",
        }

    async def remove_segment(
        self,
        *,
        track_id: str,
        audio_url: str,
        segment_start: float,
        segment_end: float,
        user_id: str | None,
        backend_id: str,
        storage_context: StorageContext,
    ) -> dict:
        validate_storage_for_backend(backend_id, storage_context)
        storage = B2Storage(storage_context)
        if segment_end <= segment_start:
            raise ServiceError(422, "segment_end must be greater than segment_start")
        result = await self._regeneration.remove_segment(
            track_id=track_id,
            audio_url=audio_url,
            segment_start=segment_start,
            segment_end=segment_end,
            user_id=user_id,
            storage=storage,
            job_id=f"remove-{uuid.uuid4()}",
        )
        return {
            "audio_url": result.audio_url,
            "b2_key": result.b2_key,
            "duration": result.duration,
            "bucket_name": result.bucket_name,
            "backend_id": backend_id,
            "segments_removed": 1,
            "removed_duration": round(segment_end - segment_start, 3),
            "track_id": track_id,
            "user_id": user_id,
            "job_type": "reconstruct",
            "action": "remove",
            "status": "completed",
        }

    async def rollback_preview(self, preview_id: str, backend_id: str) -> dict:
        if not await self._regeneration.rollback_preview(preview_id, backend_id):
            raise ServiceError(404, "preview not found or already rolled back")
        return {"preview_id": preview_id, "status": "rolled_back"}

    async def get_preview(self, preview_id: str, backend_id: str) -> dict:
        preview = await self._regeneration.get_preview(preview_id, backend_id)
        if not preview:
            raise ServiceError(404, "preview not found")
        return preview

    async def list_discovery(self, sort: str, limit: int, offset: int) -> dict:
        mode = (sort or "latest").strip().lower()
        if mode not in VALID_DISCOVERY_SORTS:
            raise ServiceError(422, "sort must be latest or trending")
        limit = max(1, min(limit or 20, 100))
        offset = max(0, offset)
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
                        "completed_at": row.completed_at.isoformat() if row.completed_at else None,
                    }
                )
            sorted_items = sort_discovery_items(items, mode)
            return {
                "sort": mode,
                "limit": limit,
                "offset": offset,
                "total": len(sorted_items),
                "items": sorted_items[offset : offset + limit],
            }
        finally:
            db.close()

    async def update_platform_settings(
        self,
        blocked_keywords: str,
        auto_tag_keywords: str,
    ) -> dict:
        blocked = [item.strip().lower() for item in blocked_keywords.split(",") if item.strip()]
        auto_tags = [
            item.strip().lower() for item in auto_tag_keywords.split(",") if item.strip()
        ]
        harm_keyword_loader.sync_platform_keywords(blocked)
        auto_tag_keyword_loader.sync(auto_tags)
        for keyword in auto_tags:
            category_loader.add_tag(keyword)
        return {
            "status": "accepted",
            "blocked_keywords_count": len(blocked),
            "auto_tag_keywords_count": len(auto_tags),
        }

    async def health(self, queue: dict) -> dict:
        return await self._health.read(queue)
