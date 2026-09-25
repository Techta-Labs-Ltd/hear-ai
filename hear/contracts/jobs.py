from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


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
    CONFIRM_PREVIEW_COMPUTE = "confirm_preview_compute"


class MagicCleanProfile(StrEnum):
    NATURAL = "natural"
    VOICE_FOCUS = "voice_focus"
    MUSIC_ATMOSPHERE = "music_atmosphere"
    STEM_MIX = "stem_mix"


class WorkerRole(StrEnum):
    PIPELINE = "pipeline"
    TRANSCRIPTION = "transcription"
    RECONSTRUCTION = "reconstruction"
    MAGIC_CLEAN_NATURAL = "magic_clean_natural"
    MAGIC_CLEAN_VOICE_FOCUS = "magic_clean_voice_focus"
    MAGIC_CLEAN_MUSIC_ATMOSPHERE = "magic_clean_music_atmosphere"
    MAGIC_CLEAN_STEM_MIX = "magic_clean_stem_mix"


class SourceReference(StrictContract):
    url: str = Field(min_length=1, max_length=4096)
    revision: int = Field(ge=1)
    file_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    pcm_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class StorageGrant(StrictContract):
    reference: str = Field(min_length=1, max_length=256)
    token: SecretStr
    expires_at: str = Field(min_length=1, max_length=128)


class ReportingGrant(StrictContract):
    backend_base_url: str = Field(min_length=1, max_length=2048)
    token: SecretStr


class AttemptEnvelope(StrictContract):
    schema_version: Literal[1] = 1
    job_id: str = Field(min_length=1, max_length=128)
    run_id: str = Field(min_length=1, max_length=128)
    attempt_id: str = Field(min_length=1, max_length=128)
    job_type: JobType
    track_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(min_length=1, max_length=128)
    source: SourceReference
    storage: StorageGrant
    reporting: ReportingGrant
    operation: ReconstructionOperation | None = None
    magic_clean_profile: MagicCleanProfile | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("job_type", mode="before")
    @classmethod
    def parse_job_type(cls, value):
        return value if isinstance(value, JobType) else JobType(value)

    @field_validator("operation", mode="before")
    @classmethod
    def parse_operation(cls, value):
        if value is None or isinstance(value, ReconstructionOperation):
            return value
        return ReconstructionOperation(value)

    @field_validator("magic_clean_profile", mode="before")
    @classmethod
    def parse_magic_clean_profile(cls, value):
        if value is None or isinstance(value, MagicCleanProfile):
            return value
        return MagicCleanProfile(value)

    @field_validator("job_id", "run_id", "attempt_id", "track_id", "user_id")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("identity value cannot be blank")
        return normalized

    @model_validator(mode="after")
    def validate_job_options(self):
        if self.job_type == JobType.RECONSTRUCTION and self.operation is None:
            raise ValueError("reconstruction requires operation")
        if self.job_type != JobType.RECONSTRUCTION and self.operation is not None:
            raise ValueError("operation is only valid for reconstruction")
        if self.job_type == JobType.MAGIC_CLEAN and self.magic_clean_profile is None:
            raise ValueError("magic_clean requires magic_clean_profile")
        if self.job_type != JobType.MAGIC_CLEAN and self.magic_clean_profile is not None:
            raise ValueError("magic_clean_profile is only valid for magic_clean")
        return self
