from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field

from .jobs import JobType, StrictContract


class ExecutionEventType(StrEnum):
    STARTED = "started"
    STAGE = "stage"
    PROGRESS = "progress"
    WARNING = "warning"
    ARTIFACT_PREPARED = "artifact_prepared"
    OUTCOME = "outcome"


class ExecutionEvent(StrictContract):
    schema_version: Literal[1] = 1
    event_id: str = Field(min_length=1, max_length=128)
    job_id: str = Field(min_length=1, max_length=128)
    run_id: str = Field(min_length=1, max_length=128)
    attempt_id: str = Field(min_length=1, max_length=128)
    track_id: str = Field(min_length=1, max_length=128)
    job_type: JobType
    source_revision: int = Field(ge=1)
    sequence: int = Field(ge=1)
    event: ExecutionEventType
    stage: str | None = Field(default=None, max_length=128)
    progress_pct: float | None = Field(default=None, ge=0, le=100)
    message: str | None = Field(default=None, max_length=512)
    metrics: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
