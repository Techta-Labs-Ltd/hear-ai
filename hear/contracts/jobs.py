from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

JobType = Literal["pipeline", "transcription", "reconstruction", "magic_clean"]
ProviderType = Literal["pod", "serverless"]


class SourceReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    url: HttpUrl
    revision: int = Field(ge=1)
    file_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    pcm_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class ReportingContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    backend_base_url: HttpUrl
    grant: str = Field(min_length=16)


class AttemptEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    job_id: str = Field(min_length=1, max_length=128)
    run_id: str = Field(min_length=1, max_length=128)
    attempt_id: str = Field(min_length=1, max_length=128)
    job_type: JobType
    operation: str = Field(default="default", min_length=1, max_length=64)
    provider: ProviderType
    track_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(min_length=1, max_length=128)
    source: SourceReference
    options: dict[str, Any] = Field(default_factory=dict)
    reporting: ReportingContext
