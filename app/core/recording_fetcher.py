from dataclasses import dataclass

import httpx

from app.config import settings


@dataclass
class TrackData:
    track_id: str
    audio_url: str
    name: str
    volume: float
    is_muted: bool
    sort_order: int
    duration: float
    is_enhanced: bool = False
    has_transcription: bool = False
    status: str = "pending"
    quality_score: float | None = None
    snr_db: float | None = None
    transcription: str | None = None
    category: str | None = None
    tags: list | None = None
    flag: dict | None = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
async def fetch_track(track_id: str) -> TrackData:
    url = f"{settings.HEAR_BACKEND_URL}/api/v1/internal/tracks/{track_id}/for-ai"
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            url,
            headers={"X-Service-Key": settings.AI_SERVICE_SECRET},
        )
        response.raise_for_status()
        data = response.json()

    return TrackData(
        track_id=data["id"],
        audio_url=data["audio_url"],
        name=data.get("name", ""),
        volume=data.get("volume", 1.0),
        is_muted=data.get("is_muted", False),
        sort_order=data.get("sort_order", 0),
        duration=data.get("duration", 0),
        is_enhanced=data.get("is_enhanced", False),
        has_transcription=data.get("transcription") is not None,
        status=data.get("status", "pending"),
        quality_score=data.get("quality_score"),
        snr_db=data.get("snr_db"),
        transcription=data.get("transcription"),
        category=data.get("category"),
        tags=data.get("tags", []),
        flag=data.get("flag"),
    )
