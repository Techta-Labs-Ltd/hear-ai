from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    ENHANCING = "enhancing"
    TRANSCRIBING = "transcribing"
    CATEGORIZING = "categorizing"
    MODERATING = "moderating"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineRequest(BaseModel):
    recording_id: str
    job_id: str
    callback_url: str
    skip_enhancement: bool = False
    skip_transcription: bool = False
    existing_transcript: Optional[str] = None
    max_tags: int = 8


class TranscribeRequest(BaseModel):
    audio_url: str
    language: Optional[str] = None
    job_id: str
    recording_id: str
    callback_url: str


class EnhanceRequest(BaseModel):
    audio_url: str
    job_id: str
    recording_id: str
    callback_url: str


class CategorizeRequest(BaseModel):
    text: str
    custom_tags: list[str] = Field(default_factory=list)
    max_tags: int = 8


class ModerateRequest(BaseModel):
    text: str


class ReconstructRequest(BaseModel):
    audio_url: str
    recording_id: str
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
