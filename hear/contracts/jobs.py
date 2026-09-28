import json
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

from hear.contracts.cleaning import CleaningProfiles
from hear.contracts.cleaning import MagicCleanProfile as MagicCleanProfile
from hear.contracts.reconstruction import ReconstructionOptions


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


class AttemptClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: ClaimDecision
    lease_seconds: float = Field(default=60.0, gt=1, le=3600)
    heartbeat_seconds: float = Field(default=15.0, gt=0, le=60)

    @model_validator(mode="after")
    def validate_intervals(self):
        if self.heartbeat_seconds >= self.lease_seconds:
            raise ValueError("heartbeat must be shorter than lease")
        return self


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
        if self.expires_at.utcoffset() is None:
            raise ValueError("storage_expiry_must_include_timezone")
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
    operation: ReconstructionOperation | None = None
    track_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(min_length=1, max_length=128)
    source: SourceReference
    storage: ArtifactStorage
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Magic Clean: natural, studio_voice, outdoor_mobile or clean_raw; optional auto_level, remove_clicks and trim_silence.",
        json_schema_extra={"examples": [{"profile": "studio_voice"}, {"profile": "clean_raw"}]},
    )
    artifact_prefix: str = Field(min_length=1, max_length=2048)
    deadline: datetime
    reporting_grant: str = Field(min_length=1, max_length=4096)
    backend_base_url: HttpUrl
    backend_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

    @property
    def workspace_namespace(self) -> str:
        return self.backend_id or "legacy"

    @model_validator(mode="after")
    def validate_operation(self):
        if self.deadline.utcoffset() is None:
            raise ValueError("deadline must include a timezone")
        if self.job_type == JobType.RECONSTRUCTION:
            if self.operation is None:
                raise ValueError("invalid reconstruction operation")
            validated = ReconstructionOptions.validate_operation(self.operation.value, self.options)
            object.__setattr__(
                self, "options", validated.model_dump(mode="json", exclude_none=True)
            )
        elif self.operation is not None:
            raise ValueError("operation is only supported for reconstruction")
        if self.job_type == JobType.MAGIC_CLEAN:
            options = self._validate_magic_clean_options(self.options)
            object.__setattr__(self, "options", options)
        if len(json.dumps(self.options, separators=(",", ":"), default=str).encode()) > 1048576:
            raise ValueError("options_too_large")
        return self

    @staticmethod
    def _validate_magic_clean_options(options: dict[str, Any]) -> dict[str, Any]:
        return CleaningProfiles.validate(options)


class WorkerIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    worker_id: str = Field(min_length=1, max_length=128)
    generation: str = Field(min_length=1, max_length=128)
    image_revision: str = Field(min_length=1, max_length=128)
    engine_revision: str = Field(min_length=1, max_length=128)
