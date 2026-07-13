import json
from dataclasses import dataclass

import httpx

from app.config import settings


def effective_transcript_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                return effective_transcript_text(json.loads(stripped))
            except Exception:
                pass
        return stripped
    if isinstance(value, dict):
        for key in ("transcript", "text", "content", "full_text", "value", "result"):
            nested = value.get(key)
            coerced = effective_transcript_text(nested)
            if coerced:
                return coerced
        return ""
    if isinstance(value, list):
        parts = [effective_transcript_text(v) for v in value]
        parts = [p for p in parts if p]
        return " ".join(parts).strip()
    return str(value).strip()


def track_payload_is_enhanced(track_payload: dict) -> bool:
    v = track_payload.get("is_enhanced")
    if v is True:
        return True
    if isinstance(v, str) and v.strip().lower() in ("true", "1", "yes"):
        return True
    if track_payload.get("enhanced_audio_url") or track_payload.get("enhanced_url"):
        return True
    return False


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
    ai_compressed_audio: dict | None = None
    ai_speed_layers: list[dict] | None = None
    ai_enhanced_audio: dict | None = None
    content_description: str | None = None
    speaker: str | None = None
    source: str | None = None
    published_at: str | None = None
    latest_at: str | None = None
    trending_score: float | None = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


async def fetch_track(track_id: str) -> TrackData:
    url = f"{settings.HEAR_BACKEND_URL}/api/v1/internal/tracks/{track_id}/for-ai"
    async with httpx.AsyncClient(timeout=5) as client:
        response = await client.get(
            url,
            headers={"X-Service-Key": settings.AI_SERVICE_SECRET},
        )
        response.raise_for_status()
        data = response.json()

    track_payload = _resolve_track_payload(data, track_id)
    tx_effective = effective_transcript_text(track_payload.get("transcription"))

    def _pair(d):
        if not isinstance(d, dict):
            return None
        bk = d.get("b2_key") if isinstance(d.get("b2_key"), str) else None
        au = d.get("audio_url") if isinstance(d.get("audio_url"), str) else None
        if not bk and not au:
            return None
        return {"b2_key": bk, "audio_url": au}

    ac = _pair(track_payload.get("ai_compressed_audio"))
    ae = _pair(track_payload.get("ai_enhanced_audio"))
    raw_layers = track_payload.get("ai_speed_layers")
    layers: list[dict] | None = None
    if isinstance(raw_layers, list):
        layers = []
        for item in raw_layers:
            if not isinstance(item, dict):
                continue
            try:
                sp = float(item["speed"])
            except (KeyError, TypeError, ValueError):
                continue
            layers.append(
                {
                    "speed": sp,
                    "b2_key": item.get("b2_key") if isinstance(item.get("b2_key"), str) else None,
                    "audio_url": item.get("audio_url") if isinstance(item.get("audio_url"), str) else None,
                }
            )
        if not layers:
            layers = None
    cd = track_payload.get("content_description")
    content_desc = cd.strip() if isinstance(cd, str) and cd.strip() else None
    sp = track_payload.get("speaker")
    speaker = sp.strip() if isinstance(sp, str) and sp.strip() else None
    src = track_payload.get("source")
    source = src.strip() if isinstance(src, str) and src.strip() else None
    pub = track_payload.get("published_at") or track_payload.get("latest_at")
    published_at = pub.strip() if isinstance(pub, str) and pub.strip() else None
    latest_at = published_at
    trend_raw = track_payload.get("trending_score")
    trending_score = None
    if trend_raw is not None:
        try:
            trending_score = float(trend_raw)
        except (TypeError, ValueError):
            trending_score = None

    return TrackData(
        track_id=track_payload.get("id", track_id),
        audio_url=track_payload.get("audio_url") or "",
        name=track_payload.get("name") or "",
        volume=track_payload.get("volume", 1.0),
        is_muted=track_payload.get("is_muted", False),
        sort_order=track_payload.get("sort_order", 0),
        duration=track_payload.get("duration", 0),
        is_enhanced=track_payload_is_enhanced(track_payload),
        has_transcription=bool(tx_effective),
        status=track_payload.get("status", "pending"),
        quality_score=track_payload.get("quality_score"),
        snr_db=track_payload.get("snr_db"),
        transcription=track_payload.get("transcription"),
        category=track_payload.get("category"),
        tags=track_payload.get("tags", []),
        flag=track_payload.get("flag"),
        ai_compressed_audio=ac,
        ai_speed_layers=layers,
        ai_enhanced_audio=ae,
        content_description=content_desc,
        speaker=speaker,
        source=source,
        published_at=published_at,
        latest_at=latest_at,
        trending_score=trending_score,
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
