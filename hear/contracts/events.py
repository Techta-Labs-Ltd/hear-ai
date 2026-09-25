from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

EventType = Literal["started", "stage", "progress", "warning", "artifact_prepared", "outcome"]


class ExecutionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    event_id: str = Field(min_length=1, max_length=128)
    job_id: str
    attempt_id: str
    track_id: str
    job_type: str
    source_revision: int = Field(ge=1)
    sequence: int = Field(ge=1)
    event: EventType
    stage: str | None = None
    progress_pct: float | None = Field(default=None, ge=0, le=100)
    message: str | None = Field(default=None, max_length=500)
    result: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
