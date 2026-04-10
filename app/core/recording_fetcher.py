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


@dataclass
class RecordingData:
    recording_id: str
    title: str
    audio_url: str
    tracks: list[TrackData]


async def fetch_recording(recording_id: str) -> RecordingData:
    url = f"{settings.HEAR_BACKEND_URL}/api/internal/recordings/{recording_id}"
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            url,
            headers={"X-Service-Key": settings.AI_SERVICE_SECRET},
        )
        response.raise_for_status()
        data = response.json()

    tracks = []
    for t in data.get("tracks", []):
        tracks.append(TrackData(
            track_id=t["id"],
            audio_url=t["audio_url"],
            name=t.get("name", ""),
            volume=t.get("volume", 1.0),
            is_muted=t.get("is_muted", False),
            sort_order=t.get("sort_order", 0),
            duration=t.get("duration", 0),
        ))

    tracks.sort(key=lambda t: t.sort_order)

    return RecordingData(
        recording_id=data["id"],
        title=data.get("title", ""),
        audio_url=data.get("audio_url", ""),
        tracks=tracks,
    )
