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

    track_payload = _resolve_track_payload(data, track_id)

    return TrackData(
        track_id=track_payload.get("id", track_id),
        audio_url=track_payload.get("audio_url") or "",
        name=track_payload.get("name") or "",
        volume=track_payload.get("volume", 1.0),
        is_muted=track_payload.get("is_muted", False),
        sort_order=track_payload.get("sort_order", 0),
        duration=track_payload.get("duration", 0),
        is_enhanced=track_payload.get("is_enhanced", False),
        has_transcription=track_payload.get("transcription") is not None,
        status=track_payload.get("status", "pending"),
        quality_score=track_payload.get("quality_score"),
        snr_db=track_payload.get("snr_db"),
        transcription=track_payload.get("transcription"),
        category=track_payload.get("category"),
        tags=track_payload.get("tags", []),
        flag=track_payload.get("flag"),
    )


def _resolve_track_payload(data: dict, requested_track_id: str) -> dict:
    if isinstance(data.get("tracks"), list):
        tracks = [t for t in data["tracks"] if isinstance(t, dict)]
        for track in tracks:
            if track.get("id") == requested_track_id:
                return track
        if isinstance(data.get("track_id"), str):
            for track in tracks:
                if track.get("id") == data["track_id"]:
                    return track
        raise ValueError(
            f"Requested track_id {requested_track_id} not found in backend payload tracks"
        )
    return data if isinstance(data, dict) else {}
