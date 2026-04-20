from dataclasses import dataclass, field

import httpx

from app.config import settings
from app.core.keyword_loader import harm_keyword_loader


@dataclass
class PlatformSettings:
    blocked_keywords: list[str] = field(default_factory=list)
    auto_tag_keywords: list[str] = field(default_factory=list)


async def fetch_platform_settings() -> PlatformSettings:
    url = f"{settings.HEAR_BACKEND_URL}/api/v1/internal/platform-settings"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(
                url,
                headers={"X-Service-Key": settings.AI_SERVICE_SECRET},
            )
            response.raise_for_status()
            data = response.json()

        blocked = [k.strip().lower() for k in data.get("blocked_keywords", "").split(",") if k.strip()]
        auto_tags = [k.strip().lower() for k in data.get("auto_tag_keywords", "").split(",") if k.strip()]

        if blocked:
            harm_keyword_loader.sync_platform_keywords(blocked)

        return PlatformSettings(
            blocked_keywords=blocked,
            auto_tag_keywords=auto_tags,
        )
    except Exception:
        return PlatformSettings()
