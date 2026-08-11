from dataclasses import dataclass, field
from hear.core.keyword_loader import auto_tag_keyword_loader, harm_keyword_loader

@dataclass
class PlatformSettings:
    blocked_keywords: list[str] = field(default_factory=list)
    auto_tag_keywords: list[str] = field(default_factory=list)

async def fetch_platform_settings() -> PlatformSettings:
    return PlatformSettings(
        blocked_keywords=harm_keyword_loader.platform_keywords,
        auto_tag_keywords=auto_tag_keyword_loader.keywords,
    )
