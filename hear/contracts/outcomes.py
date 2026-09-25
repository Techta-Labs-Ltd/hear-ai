from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .jobs import JobType


class ArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    bucket_name: str = Field(min_length=1, max_length=255)
    object_key: str = Field(min_length=1, max_length=2048)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    content_type: str = Field(min_length=1, max_length=255)
    audio_url: str | None = Field(default=None, max_length=4096)


class ExecutionOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=1, ge=1)
    job_id: str = Field(min_length=1, max_length=128)
    attempt_id: str = Field(min_length=1, max_length=128)
    track_id: str = Field(min_length=1, max_length=128)
    job_type: JobType
    source_revision: int = Field(ge=1)
    status: Literal["completed", "failed", "cancelled"]
    artifacts: tuple[ArtifactManifest, ...] = ()
    result: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = Field(default=None, max_length=128)
