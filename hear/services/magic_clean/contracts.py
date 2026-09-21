"""Cleaner v2 execution contract. Contains no transport or business persistence.

Tickets must be authenticated by the ingress before these values are used.
Schema validation is not authorization. No profile options are inferred here.
"""

from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Identity = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")]


class ErrorCode(StrEnum):
    CANCELLED = "cancelled"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    ENGINE_UNAVAILABLE = "engine_unavailable"
    SOURCE_MISMATCH = "source_mismatch"
    INVALID_AUDIO = "invalid_audio"
    PROCESS_FAILED = "process_failed"
    ARTIFACT_CONFLICT = "artifact_conflict"
    STORAGE_FAILED = "storage_failed"


class CleanExecutionError(RuntimeError):
    def __init__(self, code: ErrorCode, message: str, *, worker_restart_required: bool = False):
        super().__init__(message)
        self.code = code
        self.worker_restart_required = worker_restart_required

    def __reduce__(self):
        # Preserve typed control state across trusted Python worker/Ray boundaries.
        # Exception.args contains only the display message, not the required code.
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
    engine: Literal["deepfilternet3", "sam_audio_small", "noise_profile"]
    runtime_sha256: Digest
    checkpoint_sha256: Digest | None
    precision_policy_sha256: Digest
    longform_policy_sha256: Digest

    @model_validator(mode="after")
    def validate_checkpoint(self):
        if self.engine != "noise_profile" and self.checkpoint_sha256 is None:
            raise ValueError("model engines require a pinned checkpoint digest")
        if self.engine == "noise_profile" and self.checkpoint_sha256 is not None:
            raise ValueError("CPU noise-profile processing has no model checkpoint")
        return self


class SampleInterval(Contract):
    start_frame: int = Field(ge=0)
    end_frame: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_interval(self):
        if self.end_frame <= self.start_frame:
            raise ValueError("empty or reversed interval")
        return self


class NoiseReferenceSelection(SampleInterval):
    revision_id: Identity


class NoiseReference(NoiseReferenceSelection):
    confirmed_noise_only: bool
    analysis_sha256: Digest


class CleanPlan(Contract):
    profile: Literal["natural", "voice_focus", "music_atmosphere"]
    profile_version: Identity
    catalogue_sha256: Digest
    runtime: RuntimeIdentity
    attenuation_limit_db: Literal[12, 18, 24] | None
    noise_reduction_db: Literal[3, 6] | None
    noise_reference: NoiseReference | None
    prompt_sha256: Digest | None
    channel_policy: Literal["preserve", "mono", "validated_dual_mono"]
    mono_acknowledged: bool
    adjust_loudness: bool
    match_comparison_loudness: Literal[True]
    shorten_pauses: Literal[False]
    seed: int = Field(ge=0, le=2**63 - 1)

    @model_validator(mode="after")
    def validate_profile(self):
        expected = {
            "natural": "deepfilternet3",
            "voice_focus": "sam_audio_small",
            "music_atmosphere": "noise_profile",
        }
        if self.runtime.engine != expected[self.profile]:
            raise ValueError("profile and engine do not match")
        if self.profile == "natural":
            if self.attenuation_limit_db is None or self.channel_policy != "preserve":
                raise ValueError("Natural requires explicit attenuation and preserved channels")
        elif self.attenuation_limit_db is not None:
            raise ValueError("attenuation limit is only supported by Natural")
        if self.profile == "music_atmosphere":
            if self.noise_reduction_db is None or self.noise_reference is None:
                raise ValueError(
                    "Music & Atmosphere requires an explicit noise reference/reduction"
                )
            if not self.noise_reference.confirmed_noise_only or self.channel_policy != "preserve":
                raise ValueError("confirmed reference and preserved channels are required")
        elif self.noise_reduction_db is not None or self.noise_reference is not None:
            raise ValueError("noise references are only supported by Music & Atmosphere")
        if self.profile == "voice_focus":
            if self.prompt_sha256 is None or self.channel_policy == "preserve":
                raise ValueError("Voice Focus requires pinned prompt and explicit mono policy")
        elif self.prompt_sha256 is not None:
            raise ValueError("prompt conditioning is only supported by Voice Focus")
        return self


class AttemptTicket(Contract):
    contract_version: Literal["hear.cleaner.v2"]
    backend_id: Identity
    tenant_scope: Identity
    job_id: Identity
    attempt_id: Identity
    fence: int = Field(gt=0)
    provider: Literal["pod", "serverless"]
    purpose: Literal["sample_preview", "full_candidate"]
    input: SourceIdentity
    expected_active_audio_revision: Identity
    plan: CleanPlan
    sample: SampleInterval | None
    artifact_prefix: str
    manifest_key: str
    deadline: datetime
    heartbeat_seconds: int = Field(ge=1, le=60)
    lease_seconds: int = Field(gt=1, le=3600)
    correlation_id: Identity

    @field_validator("deadline")
    @classmethod
    def aware_deadline(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("deadline requires a timezone")
        return value

    @model_validator(mode="after")
    def validate_attempt(self):
        parts = self.artifact_prefix.split("/")
        if any(part in ("", ".", "..") for part in parts) or "\\" in self.artifact_prefix:
            raise ValueError("artifact prefix must be a canonical relative object path")
        if self.attempt_id not in parts or self.job_id not in parts or self.backend_id not in parts:
            raise ValueError("artifact prefix must identify backend, job and attempt")
        if self.manifest_key != str(PurePosixPath(self.artifact_prefix) / "manifest.json"):
            raise ValueError("manifest must be inside the attempt prefix")
        if self.lease_seconds <= self.heartbeat_seconds:
            raise ValueError("lease must exceed heartbeat interval")
        if (self.purpose == "sample_preview") != (self.sample is not None):
            raise ValueError("only sample previews require a sample interval")
        if self.sample and self.sample.end_frame > self.input.frames:
            raise ValueError("sample exceeds the pinned source")
        reference = self.plan.noise_reference
        if reference and (
            reference.revision_id != self.input.revision_id
            or reference.end_frame > self.input.frames
        ):
            raise ValueError("noise reference must belong to the actual pinned input")
        if self.plan.profile == "voice_focus":
            if self.input.channels == 2 and (
                not self.plan.mono_acknowledged or self.plan.channel_policy != "validated_dual_mono"
            ):
                raise ValueError("stereo Voice Focus requires acknowledged, validated dual mono")
        return self


class StorageGrant(Contract):
    """Separate from semantic identity; SecretStr prevents accidental logging."""

    reference: Identity
    token: SecretStr
    expires_at: datetime

    @field_validator("expires_at")
    @classmethod
    def aware_expiry(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("grant expiry requires a timezone")
        return value


class ArtifactIdentity(Contract):
    role: Literal[
        "cleaned_master", "delivery_audio", "comparison_source", "edit_map", "validation_report"
    ]
    object_key: str = Field(min_length=1, max_length=2048)
    object_version: str = Field(min_length=1, max_length=1024)
    sha256: Digest
    size_bytes: int = Field(gt=0)
    content_type: Literal["audio/flac", "audio/mpeg", "application/json"]


class ContentWarningInterval(SampleInterval):
    """Coarse signal-risk interval on the pinned source frame grid, not VAD proof."""

    code: Literal["possible_wanted_content_loss", "no_speech_target_detected"]
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


class CleanResultManifest(Contract):
    contract_version: Literal["hear.cleaner.result.v2"]
    backend_id: Identity
    tenant_scope: Identity
    job_id: Identity
    attempt_id: Identity
    fence: int = Field(gt=0)
    purpose: Literal["sample_preview", "full_candidate"]
    source: SourceIdentity
    expected_active_audio_revision: Identity
    plan: CleanPlan
    plan_sha256: Digest
    sample: SampleInterval | None
    deadline: datetime
    completed_at: datetime
    outcome: Literal["succeeded", "failed", "cancelled"]
    error_code: ErrorCode | None
    validation: ValidationSummary
    artifacts: tuple[ArtifactIdentity, ...]
    timing: Literal["identity", "sample_identity"]
    correlation_id: Identity

    @model_validator(mode="after")
    def validate_result(self):
        if self.deadline.utcoffset() is None or self.completed_at.utcoffset() is None:
            raise ValueError("result timestamps require a timezone")
        if any(
            interval.end_frame > self.source.frames
            for interval in self.validation.source_warning_intervals
        ):
            raise ValueError("warning interval exceeds pinned source")
        speech = self.validation.speech_activity
        if speech:
            output_channels = 1 if self.plan.profile == "voice_focus" else self.source.channels
            if (
                speech.source_sha256 != self.source.sha256
                or len(speech.source_active_frames) != self.source.channels
                or len(speech.output_active_frames) != output_channels
                or any(
                    v > self.source.frames
                    for v in (*speech.source_active_frames, *speech.output_active_frames)
                )
                or any(
                    v.end_frame > self.source.frames or v.channel >= self.source.channels
                    for v in speech.source_loss_intervals
                )
            ):
                raise ValueError("speech evidence does not match pinned source/layout")
        if len({artifact.role for artifact in self.artifacts}) != len(self.artifacts):
            raise ValueError("duplicate artifact role")
        if len({artifact.object_key for artifact in self.artifacts}) != len(self.artifacts):
            raise ValueError("duplicate artifact object")
        if (self.purpose == "sample_preview") != (self.sample is not None):
            raise ValueError("sample identity does not match purpose")
        expected_timing = "sample_identity" if self.sample else "identity"
        if self.timing != expected_timing:
            raise ValueError("timing does not match purpose")
        if self.outcome == "succeeded":
            if self.error_code is not None or self.completed_at >= self.deadline:
                raise ValueError("successful result requires no error and a live deadline")
            if self.validation.hard_integrity != "passed":
                raise ValueError("successful result requires hard integrity to pass")
            if self.validation.wanted_content in ("rejected", "not_applicable"):
                raise ValueError("successful result requires wanted-content checks")
            required = {"cleaned_master", "delivery_audio", "validation_report"}
            if not required.issubset({artifact.role for artifact in self.artifacts}):
                raise ValueError("successful result is missing required artifacts")
        elif self.error_code is None or self.artifacts:
            raise ValueError("failed/cancelled results require an error and no candidate artifacts")
        elif (self.outcome == "cancelled") != (self.error_code == ErrorCode.CANCELLED):
            raise ValueError("cancelled outcome requires the cancelled error code")
        elif (
            self.validation.hard_integrity != "not_applicable"
            or self.validation.wanted_content != "not_applicable"
        ):
            raise ValueError("failed/cancelled results cannot claim candidate validation")
        return self


class TerminalReference(Contract):
    """Compact internal notice; backend must verify the referenced bundle."""

    backend_id: Identity
    tenant_scope: Identity
    job_id: Identity
    attempt_id: Identity
    fence: int = Field(gt=0)
    object_key: str = Field(min_length=1, max_length=2048)
    object_version: str = Field(min_length=1, max_length=1024)
    sha256: Digest
    size_bytes: int = Field(gt=0, le=256 * 1024)
    outcome: Literal["succeeded", "failed", "cancelled"]
    error_code: ErrorCode | None

    @model_validator(mode="after")
    def validate_outcome(self):
        if (self.outcome == "succeeded") != (self.error_code is None):
            raise ValueError("terminal outcome/error mismatch")
        if (self.outcome == "cancelled") != (self.error_code == ErrorCode.CANCELLED):
            raise ValueError("cancelled outcome/error mismatch")
        return self
