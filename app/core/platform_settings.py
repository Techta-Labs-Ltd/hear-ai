from dataclasses import dataclass, field

import httpx

from app.config import settings


@dataclass
class PlatformSettings:
    blocked_keywords: list[str] = field(default_factory=list)
    auto_tag_keywords: list[str] = field(default_factory=list)


async def fetch_platform_settings(organization_id: str = "") -> PlatformSettings:
    url = f"{settings.HEAR_BACKEND_URL}/api/internal/platform-settings"
    params = {}
    if organization_id:
        params["organization_id"] = organization_id

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(
                url,
                headers={"X-Service-Key": settings.AI_SERVICE_SECRET},
                params=params,
            )
            response.raise_for_status()
            data = response.json()

        blocked_raw = data.get("blocked_keywords", "")
        auto_tag_raw = data.get("auto_tag_keywords", "")

        blocked = [k.strip().lower() for k in blocked_raw.split(",") if k.strip()]
        auto_tags = [k.strip().lower() for k in auto_tag_raw.split(",") if k.strip()]

        return PlatformSettings(
            blocked_keywords=blocked,
            auto_tag_keywords=auto_tags,
        )
    except Exception:
        return PlatformSettings()
