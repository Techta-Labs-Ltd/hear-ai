from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PipelineRequest(BaseModel):
    job_id: str
    track_id: str
    job_type: str = "pipeline"
    max_tags: int = 8
    edited_transcript: Optional[str] = None
    grouped: bool = False
    group_id: Optional[str] = None
    kind: str = "track"
    source: Optional[str] = None
    track_count: int = 1


class RealtimeRequest(BaseModel):
    job_id: str
    track_id: str
    job_type: str = "pipeline"
    max_tags: int = 8
    grouped: bool = False
    group_id: Optional[str] = None
    kind: str = "track"
    source: Optional[str] = None
    track_count: int = 1


class TranscribeRequest(BaseModel):
    job_id: str
    track_id: str


class EnhanceRequest(BaseModel):
    job_id: str
    track_id: str


class CategorizeRequest(BaseModel):
    text: str
    custom_tags: list[str] = Field(default_factory=list)
    max_tags: int = 8


class ModerateRequest(BaseModel):
    text: str


class ReconstructRequest(BaseModel):
    audio_url: str
    track_id: str
    segment_start: float
    segment_end: float
    new_text: str


class JobAccepted(BaseModel):
    job_id: str
    status: str = "accepted"


class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    gpu_name: str
    models_loaded: list[str]
    active_jobs: int
    queued_jobs: int
