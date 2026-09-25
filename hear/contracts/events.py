from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .jobs import JobType


class ExecutionEventType(StrEnum):
    STARTED = "started"
    STAGE = "stage"
    PROGRESS = "progress"
    WARNING = "warning"
    ARTIFACT_PREPARED = "artifact_prepared"
    OUTCOME = "outcome"


class ExecutionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=1, ge=1)
    event_id: str = Field(min_length=1, max_length=128)
    job_id: str = Field(min_length=1, max_length=128)
    attempt_id: str = Field(min_length=1, max_length=128)
    track_id: str = Field(min_length=1, max_length=128)
    job_type: JobType
    source_revision: int = Field(ge=1)
    sequence: int = Field(ge=1)
    event: ExecutionEventType
    stage: str | None = Field(default=None, max_length=64)
    progress_pct: float | None = Field(default=None, ge=0, le=100)
    message: str | None = Field(default=None, max_length=512)
    data: dict[str, Any] = Field(default_factory=dict)
