from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    bucket_name: str
    object_key: str
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    content_type: str
    audio_url: str | None = None


class ExecutionOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["completed", "failed", "cancelled"]
    manifest: ArtifactManifest | None = None
    error_code: str | None = None
