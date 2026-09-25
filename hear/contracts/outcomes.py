from enum import StrEnum
from typing import Any, Literal

from pydantic import Field

from .jobs import JobType, StrictContract


class OutcomeStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ArtifactManifest(StrictContract):
    bucket_name: str = Field(min_length=1, max_length=256)
    object_key: str = Field(min_length=1, max_length=2048)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    content_type: str = Field(min_length=1, max_length=256)
    source_revision: int = Field(ge=1)
    attempt_id: str = Field(min_length=1, max_length=128)
    engine_revision: str = Field(min_length=1, max_length=256)


class JobOutcome(StrictContract):
    schema_version: Literal[1] = 1
    event_id: str = Field(min_length=1, max_length=128)
    job_id: str = Field(min_length=1, max_length=128)
    run_id: str = Field(min_length=1, max_length=128)
    attempt_id: str = Field(min_length=1, max_length=128)
    track_id: str = Field(min_length=1, max_length=128)
    job_type: JobType
    source_revision: int = Field(ge=1)
    status: OutcomeStatus
    artifacts: tuple[ArtifactManifest, ...] = ()
    result: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = Field(default=None, max_length=128)
    error_message: str | None = Field(default=None, max_length=512)
