"""Cleaning engine contracts: plan, runtime identity, and validation evidence.

Schema validation is not authorization; options arrive validated by
``CleaningProfiles`` and the engine identity is pinned by the loader.
"""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Identity = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")]


class ErrorCode(StrEnum):
    CANCELLED = "cancelled"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    ENGINE_UNAVAILABLE = "engine_unavailable"
    SOURCE_MISMATCH = "source_mismatch"
    INVALID_AUDIO = "invalid_audio"
    TARGET_NOT_DETECTED = "target_not_detected"
    PROCESS_FAILED = "process_failed"
    ARTIFACT_CONFLICT = "artifact_conflict"
    STORAGE_FAILED = "storage_failed"


class CleanExecutionError(RuntimeError):
    def __init__(self, code: ErrorCode, message: str, *, worker_restart_required: bool = False):
        super().__init__(message)
        self.code = code
        self.worker_restart_required = worker_restart_required

    def __reduce__(self):
        return type(self), (self.code, str(self)), self.__dict__


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class SourceIdentity(Contract):
    revision_id: Identity
    media_id: Identity
    object_key: str = Field(min_length=1, max_length=2048)
    object_version: str = Field(min_length=1, max_length=1024)
    sha256: Digest
    size_bytes: int = Field(gt=0)
    sample_rate: int = Field(ge=8000, le=96000)
    channels: Literal[1, 2]
    frames: int = Field(gt=0)


class RuntimeIdentity(Contract):
    engine: Literal["deepfilternet3"]
    runtime_sha256: Digest
    checkpoint_sha256: Digest | None
    precision_policy_sha256: Digest
    longform_policy_sha256: Digest

    @model_validator(mode="after")
    def validate_checkpoint(self):
        if self.checkpoint_sha256 is None:
            raise ValueError("model engines require a pinned checkpoint digest")
        return self


class SampleInterval(Contract):
    start_frame: int = Field(ge=0)
    end_frame: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_interval(self):
        if self.end_frame <= self.start_frame:
            raise ValueError("empty or reversed interval")
        return self


class CleanPlan(Contract):
    profile: Literal["natural"]
    profile_version: Identity
    catalogue_sha256: Digest
    runtime: RuntimeIdentity
    attenuation_limit_db: Annotated[int, Field(ge=6, le=60)] | None
    post_filter: bool = False
    prompt_sha256: Digest | None
    prompt_text: str | None = Field(default=None, min_length=1, max_length=160)
    prompt_action: Literal["isolate", "remove"] = "isolate"
    prompt_mode: Literal["ambient", "event"] | None = None
    channel_policy: Literal["preserve", "mono", "validated_dual_mono"]
    mono_acknowledged: bool
    adjust_loudness: bool
    match_comparison_loudness: Literal[True]
    shorten_pauses: Literal[False]
    seed: int = Field(ge=0, le=2**63 - 1)

    @model_validator(mode="after")
    def validate_profile(self):
        if self.attenuation_limit_db is None or self.channel_policy != "preserve":
            raise ValueError("Natural requires attenuation and preserved channels")
        if (
            self.prompt_sha256 is not None
            or self.prompt_text is not None
            or self.prompt_action != "isolate"
            or self.prompt_mode is not None
        ):
            raise ValueError("prompt-conditioned separation is no longer supported")
        return self


class ContentWarningInterval(SampleInterval):
    """Coarse signal-risk interval on the pinned source frame grid, not VAD proof."""

    code: Literal["possible_wanted_content_loss"]
    minimum_rms_ratio: float = Field(ge=0, le=1)


class SpeechLossInterval(SampleInterval):
    channel: int = Field(ge=0, le=1)


class SpeechRiskEvidence(Contract):
    source_sha256: Digest
    output_sha256: Digest
    analysis_sha256: Digest
    comparison_sha256: Digest
    source_active_frames: tuple[Annotated[int, Field(ge=0)], ...] = Field(
        min_length=1, max_length=2
    )
    output_active_frames: tuple[Annotated[int, Field(ge=0)], ...] = Field(
        min_length=1, max_length=2
    )
    source_loss_intervals: tuple[SpeechLossInterval, ...] = Field(default=(), max_length=128)
    evidence_truncated: bool


class ValidationSummary(Contract):
    hard_integrity: Literal["passed", "rejected", "not_applicable"]
    wanted_content: Literal["passed", "review_required", "rejected", "not_applicable"]
    warning_codes: tuple[Identity, ...]
    source_warning_intervals: tuple[ContentWarningInterval, ...] = Field(default=(), max_length=128)
    warning_intervals_truncated: bool = False
    speech_activity: SpeechRiskEvidence | None = None

    @model_validator(mode="after")
    def validate_intervals(self):
        if any(
            interval.code not in self.warning_codes for interval in self.source_warning_intervals
        ):
            raise ValueError("warning interval has no summary code")
        if self.hard_integrity == "not_applicable" and (
            self.source_warning_intervals
            or self.warning_intervals_truncated
            or self.speech_activity
        ):
            raise ValueError("unvalidated results cannot include signal-risk intervals")
        if self.speech_activity:
            if (
                self.speech_activity.source_loss_intervals
                and "possible_speech_loss" not in self.warning_codes
            ):
                raise ValueError("speech loss intervals require a warning")
            if (
                self.speech_activity.evidence_truncated
                and "speech_evidence_incomplete" not in self.warning_codes
            ):
                raise ValueError("truncated speech evidence requires a warning")
        return self
