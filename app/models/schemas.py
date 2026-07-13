from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


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


class PipelineRequest(BaseModel):
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


class TaxonomySyncResponse(BaseModel):
    tags_added: list[str] = Field(default_factory=list)
    categories_added: list[str] = Field(default_factory=list)
    total_tags: int = 0
    total_categories: int = 0


class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    gpu_name: str
    models_loaded: list[str]
    active_jobs: int
    queued_jobs: int
    gpu_memory: dict[str, float] = {}
    redis_status: str = "disabled"
