from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, model_validator


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SegmentChange(BaseModel):
    segment_start: float
    segment_end: float
    new_text: str
    original_text: Optional[str] = None


class StorageContext(BaseModel):
    endpoint_url: str = Field(min_length=1)
    bucket_name: str = Field(min_length=1)
    key_id: str = Field(min_length=1)
    application_key: str = Field(min_length=1)
    folder_prefix: str = Field(min_length=1)
    public_base_url: str = Field(min_length=1)
    expires_at: datetime

    @model_validator(mode="after")
    def validate_storage_context(self):
        for name in ("endpoint_url", "public_base_url"):
            parsed = urlsplit(getattr(self, name))
            if parsed.scheme != "https" or not parsed.netloc or parsed.username:
                raise ValueError(f"{name} must be an absolute HTTPS URL")
            if parsed.query or parsed.fragment:
                raise ValueError(f"{name} must not contain query or fragment")
        prefix = self.folder_prefix.strip().strip("/")
        parts = prefix.split("/")
        if not prefix or any(part in {"", ".", ".."} for part in parts):
            raise ValueError("folder_prefix must be a safe relative object prefix")
        if any("\\" in part or "\x00" in part for part in parts):
            raise ValueError("folder_prefix contains unsafe characters")
        self.folder_prefix = prefix + "/"
        expires = self.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        else:
            expires = expires.astimezone(timezone.utc)
        self.expires_at = expires
        return self


class PipelineRequest(BaseModel):
    backend_id: str = Field(min_length=1)
    storage: StorageContext
    job_id: str
    track_id: str
    job_type: str = "pipeline"
    max_tags: int = 8
    edited_transcript: Optional[str] = None
    audio_url: Optional[str] = None
    changes: list[SegmentChange] = Field(default_factory=list)
    segment_start: Optional[float] = None
    segment_end: Optional[float] = None
    new_text: Optional[str] = None
    same_speaker: bool = True
    grouped: bool = False
    group_id: Optional[str] = None
    kind: str = "track"
    source: Optional[str] = None
    track_count: int = 1
    speed_multipliers: Optional[list[float]] = None
    playback_instruction: Optional[str] = None
    type: Optional[str] = None
    media_file_id: Optional[str] = None
    user_id: str = Field(min_length=1)
    speech: Optional[int] = Field(default=None, ge=0, le=100)
    music: Optional[int] = Field(default=None, ge=0, le=100)
    background: Optional[int] = Field(default=None, ge=0, le=100)
    cut_silence: bool = False

    @model_validator(mode="after")
    def validate_stem_levels(self):
        levels = (self.speech, self.music, self.background)
        if any(value is not None for value in levels) and not all(
            value is not None for value in levels
        ):
            raise ValueError("speech, music, and background must be supplied together")
        return self


class ProcessResponse(BaseModel):
    backend_id: str
    job_id: str
    run_id: str
    track_id: str
    job_type: str
    status: str
    replayed: bool


class DiscoveryProcessRequest(BaseModel):
    backend_id: str = Field(min_length=1)
    storage: StorageContext
    job_id: str
    track_id: str
    audio_url: Optional[str] = None
    user_id: str = Field(min_length=1)
    source: Optional[str] = None


class RealtimeRequest(BaseModel):
    job_id: str
    track_id: str
    job_type: str = "pipeline"
    max_tags: int = 8
    audio_url: Optional[str] = None
    changes: list[SegmentChange] = Field(default_factory=list)
    segment_start: Optional[float] = None
    segment_end: Optional[float] = None
    new_text: Optional[str] = None
    same_speaker: bool = True
    grouped: bool = False
    group_id: Optional[str] = None
    kind: str = "track"
    source: Optional[str] = None
    track_count: int = 1
    speed_multipliers: Optional[list[float]] = None
    playback_instruction: Optional[str] = None
    type: Optional[str] = None
    media_file_id: Optional[str] = None


class TranscribeRequest(BaseModel):
    job_id: str
    track_id: str


class EnhanceRequest(BaseModel):
    job_id: str
    track_id: str


class CategorizeResponse(BaseModel):
    tags: list[str]
    categories: list[str]
    confidence_scores: dict[str, float] = Field(default_factory=dict)
    sentiment: str = "neutral"
    new_tags_added: list[str] = Field(default_factory=list)
    new_categories_added: list[str] = Field(default_factory=list)
    settings_applied: bool = False
    llm_used: bool = False
    categorizer_mode: str = "nli"


class CategorizeRequest(BaseModel):
    text: str
    custom_tags: list[str] = Field(default_factory=list)
    max_tags: int = 8


class ModerateRequest(BaseModel):
    text: str


class ReconstructRequest(BaseModel):
    audio_url: str
    track_id: str
    changes: list[SegmentChange] = Field(default_factory=list)
    segment_start: Optional[float] = None
    segment_end: Optional[float] = None
    new_text: Optional[str] = None
    same_speaker: bool = True


class EditTranscriptRequest(BaseModel):
    job_id: str
    track_id: str
    edited_transcript: str
    same_speaker: bool = True
    user_id: Optional[str] = None


class ReconstructConfirmRequest(BaseModel):
    preview_id: str
    track_id: str
    user_id: Optional[str] = None


class ReconstructRemoveRequest(BaseModel):
    track_id: str
    audio_url: str
    segment_start: float
    segment_end: float
    user_id: Optional[str] = None


class JobAccepted(BaseModel):
    job_id: str
    status: str = "accepted"


class DiscoveryCatalogItem(BaseModel):
    track_id: str
    job_id: str
    discovery: dict
    latest_at: str = ""
    published_at: str = ""
    trending_score: float = 0.0
    completed_at: str | None = None


class DiscoveryCatalogResponse(BaseModel):
    sort: str
    limit: int
    offset: int
    total: int
    items: list[DiscoveryCatalogItem]


class CategoryWebhookEvent(BaseModel):
    event_type: str
    text: str
    category: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    label: Optional[str] = None
    source_id: Optional[str] = None


class CategoryWebhookResponse(BaseModel):
    status: str = "accepted"
    example_id: str


class TrainCategorizerResponse(BaseModel):
    status: str
    detail: str = ""


class PlatformSettingsWebhookEvent(BaseModel):
    blocked_keywords: str = ""    # comma-separated, e.g. "spam,scam,fraud"
    auto_tag_keywords: str = ""   # comma-separated, e.g. "news,breaking,exclusive,interview,report"


class PlatformSettingsWebhookResponse(BaseModel):
    status: str = "accepted"
    blocked_keywords_count: int = 0
    auto_tag_keywords_count: int = 0


class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    gpu_name: str
    models_loaded: list[str]
    active_jobs: int
    queued_jobs: int
    gpu_memory: dict[str, float] = {}
    cache_backend: str = "ray_actor"
