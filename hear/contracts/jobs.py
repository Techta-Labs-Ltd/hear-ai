from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    SecretStr,
    field_serializer,
    model_validator,
)


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


class ClaimDecision(StrEnum):
    EXECUTE = "execute"
    ALREADY_COMPLETED = "already_completed"
    CANCELLED = "cancelled"
    STALE = "stale"
    NOT_CURRENT = "not_current"
    LEASE_UNAVAILABLE = "lease_unavailable"


class MagicCleanProfile(StrEnum):
    NATURAL = "natural"
    VOICE_FOCUS = "voice_focus"
    MUSIC_ATMOSPHERE = "music_atmosphere"
    STEM_MIX = "stem_mix"


class ArtifactStorage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    endpoint_url: HttpUrl
    bucket_name: str = Field(min_length=1, max_length=255)
    key_id: str = Field(min_length=1, max_length=255)
    application_key: SecretStr
    folder_prefix: str = Field(min_length=1, max_length=2048)
    public_base_url: HttpUrl
    expires_at: datetime

    @model_validator(mode="after")
    def validate_prefix(self):
        prefix = self.folder_prefix.strip().strip("/")
        parts = prefix.split("/")
        if not prefix or any(part in {"", ".", ".."} for part in parts):
            raise ValueError("invalid storage folder prefix")
        if any("\\" in part or "\x00" in part for part in parts):
            raise ValueError("invalid storage folder prefix")
        object.__setattr__(self, "folder_prefix", prefix + "/")
        return self

    @field_serializer("application_key", when_used="json")
    def serialize_application_key(self, value: SecretStr) -> str:
        return value.get_secret_value()


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
    storage: ArtifactStorage
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


class WorkerIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    worker_id: str = Field(min_length=1, max_length=128)
    generation: str = Field(min_length=1, max_length=128)
    image_revision: str = Field(min_length=1, max_length=128)
    engine_revision: str = Field(min_length=1, max_length=128)