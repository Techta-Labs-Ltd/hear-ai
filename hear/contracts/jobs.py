from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class JobType(StrEnum):
    PIPELINE = "pipeline"
    TRANSCRIPTION = "transcription"
    RECONSTRUCTION = "reconstruction"
    MAGIC_CLEAN = "magic_clean"


class ReconstructionOperation(StrEnum):
    REPLACE_SEGMENTS = "replace_segments"
    EDIT_TRANSCRIPT = "edit_transcript"
    REBUILD = "rebuild"
    REMOVE_SEGMENTS = "remove_segments"
    PREVIEW = "preview"


class MagicCleanProfile(StrEnum):
    NATURAL = "natural"
    VOICE_FOCUS = "voice_focus"
    MUSIC_ATMOSPHERE = "music_atmosphere"
    STEM_MIX = "stem_mix"


class SourceReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    url: HttpUrl
    revision: int = Field(ge=1)
    file_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    pcm_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class AttemptEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=1, ge=1)
    job_id: str = Field(min_length=1, max_length=128)
    run_id: str = Field(min_length=1, max_length=128)
    attempt_id: str = Field(min_length=1, max_length=128)
    job_type: JobType
    operation: str | None = Field(default=None, max_length=64)
    track_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(min_length=1, max_length=128)
    source: SourceReference
    options: dict[str, Any] = Field(default_factory=dict)
    artifact_prefix: str = Field(min_length=1, max_length=2048)
    deadline: datetime
    reporting_grant: str = Field(min_length=1)
    backend_base_url: HttpUrl

    @model_validator(mode="after")
    def validate_operation(self):
        if self.job_type == JobType.RECONSTRUCTION:
            if self.operation not in {item.value for item in ReconstructionOperation}:
                raise ValueError("invalid reconstruction operation")
        elif self.operation not in (None, "", "default"):
            raise ValueError("operation is only supported for reconstruction")
        return self
