import json
from dataclasses import dataclass


def effective_transcript_text(value) -> str:
    """Normalize supported transcription result shapes to plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        if stripped.startswith(("{", "[")):
            try:
                return effective_transcript_text(json.loads(stripped))
            except (json.JSONDecodeError, TypeError):
                pass
        return stripped
    if isinstance(value, dict):
        for key in ("transcript", "text", "content", "full_text", "value", "result"):
            text = effective_transcript_text(value.get(key))
            if text:
                return text
        return ""
    if isinstance(value, list):
        return " ".join(filter(None, map(effective_transcript_text, value))).strip()
    return str(value).strip()


@dataclass(slots=True)
class TrackData:
    """Minimal processing context built exclusively from a submitted job."""

    track_id: str
    audio_url: str
    name: str = ""
    duration: float = 0
    transcription: str | None = None
    has_transcription: bool = False
    content_description: str | None = None
    speaker: str | None = None
    source: str | None = None
    category: str | None = None
    published_at: str | None = None
    trending_score: float | None = None
