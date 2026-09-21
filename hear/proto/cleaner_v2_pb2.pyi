from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class RuntimeIdentity(_message.Message):
    __slots__ = ("engine", "runtime_sha256", "checkpoint_sha256", "precision_policy_sha256", "longform_policy_sha256")
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_SHA256_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_SHA256_FIELD_NUMBER: _ClassVar[int]
    PRECISION_POLICY_SHA256_FIELD_NUMBER: _ClassVar[int]
    LONGFORM_POLICY_SHA256_FIELD_NUMBER: _ClassVar[int]
    engine: str
    runtime_sha256: str
    checkpoint_sha256: str
    precision_policy_sha256: str
    longform_policy_sha256: str
    def __init__(self, engine: _Optional[str] = ..., runtime_sha256: _Optional[str] = ..., checkpoint_sha256: _Optional[str] = ..., precision_policy_sha256: _Optional[str] = ..., longform_policy_sha256: _Optional[str] = ...) -> None: ...

class SourceIdentity(_message.Message):
    __slots__ = ("revision_id", "media_id", "object_key", "object_version", "sha256", "size_bytes", "sample_rate", "channels", "frames")
    REVISION_ID_FIELD_NUMBER: _ClassVar[int]
    MEDIA_ID_FIELD_NUMBER: _ClassVar[int]
    OBJECT_KEY_FIELD_NUMBER: _ClassVar[int]
    OBJECT_VERSION_FIELD_NUMBER: _ClassVar[int]
    SHA256_FIELD_NUMBER: _ClassVar[int]
    SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    CHANNELS_FIELD_NUMBER: _ClassVar[int]
    FRAMES_FIELD_NUMBER: _ClassVar[int]
    revision_id: str
    media_id: str
    object_key: str
    object_version: str
    sha256: str
    size_bytes: int
    sample_rate: int
    channels: int
    frames: int
    def __init__(self, revision_id: _Optional[str] = ..., media_id: _Optional[str] = ..., object_key: _Optional[str] = ..., object_version: _Optional[str] = ..., sha256: _Optional[str] = ..., size_bytes: _Optional[int] = ..., sample_rate: _Optional[int] = ..., channels: _Optional[int] = ..., frames: _Optional[int] = ...) -> None: ...

class SampleInterval(_message.Message):
    __slots__ = ("start_frame", "end_frame")
    START_FRAME_FIELD_NUMBER: _ClassVar[int]
    END_FRAME_FIELD_NUMBER: _ClassVar[int]
    start_frame: int
    end_frame: int
    def __init__(self, start_frame: _Optional[int] = ..., end_frame: _Optional[int] = ...) -> None: ...

class NoiseReference(_message.Message):
    __slots__ = ("start_frame", "end_frame", "revision_id", "confirmed_noise_only", "analysis_sha256")
    START_FRAME_FIELD_NUMBER: _ClassVar[int]
    END_FRAME_FIELD_NUMBER: _ClassVar[int]
    REVISION_ID_FIELD_NUMBER: _ClassVar[int]
    CONFIRMED_NOISE_ONLY_FIELD_NUMBER: _ClassVar[int]
    ANALYSIS_SHA256_FIELD_NUMBER: _ClassVar[int]
    start_frame: int
    end_frame: int
    revision_id: str
    confirmed_noise_only: bool
    analysis_sha256: str
    def __init__(self, start_frame: _Optional[int] = ..., end_frame: _Optional[int] = ..., revision_id: _Optional[str] = ..., confirmed_noise_only: bool = ..., analysis_sha256: _Optional[str] = ...) -> None: ...

class CleanPlan(_message.Message):
    __slots__ = ("profile", "profile_version", "catalogue_sha256", "runtime", "attenuation_limit_db", "noise_reduction_db", "noise_reference", "prompt_sha256", "channel_policy", "mono_acknowledged", "adjust_loudness", "match_comparison_loudness", "shorten_pauses", "seed")
    PROFILE_FIELD_NUMBER: _ClassVar[int]
    PROFILE_VERSION_FIELD_NUMBER: _ClassVar[int]
    CATALOGUE_SHA256_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_FIELD_NUMBER: _ClassVar[int]
    ATTENUATION_LIMIT_DB_FIELD_NUMBER: _ClassVar[int]
    NOISE_REDUCTION_DB_FIELD_NUMBER: _ClassVar[int]
    NOISE_REFERENCE_FIELD_NUMBER: _ClassVar[int]
    PROMPT_SHA256_FIELD_NUMBER: _ClassVar[int]
    CHANNEL_POLICY_FIELD_NUMBER: _ClassVar[int]
    MONO_ACKNOWLEDGED_FIELD_NUMBER: _ClassVar[int]
    ADJUST_LOUDNESS_FIELD_NUMBER: _ClassVar[int]
    MATCH_COMPARISON_LOUDNESS_FIELD_NUMBER: _ClassVar[int]
    SHORTEN_PAUSES_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    profile: str
    profile_version: str
    catalogue_sha256: str
    runtime: RuntimeIdentity
    attenuation_limit_db: int
    noise_reduction_db: int
    noise_reference: NoiseReference
    prompt_sha256: str
    channel_policy: str
    mono_acknowledged: bool
    adjust_loudness: bool
    match_comparison_loudness: bool
    shorten_pauses: bool
    seed: int
    def __init__(self, profile: _Optional[str] = ..., profile_version: _Optional[str] = ..., catalogue_sha256: _Optional[str] = ..., runtime: _Optional[_Union[RuntimeIdentity, _Mapping]] = ..., attenuation_limit_db: _Optional[int] = ..., noise_reduction_db: _Optional[int] = ..., noise_reference: _Optional[_Union[NoiseReference, _Mapping]] = ..., prompt_sha256: _Optional[str] = ..., channel_policy: _Optional[str] = ..., mono_acknowledged: bool = ..., adjust_loudness: bool = ..., match_comparison_loudness: bool = ..., shorten_pauses: bool = ..., seed: _Optional[int] = ...) -> None: ...

class AttemptTicket(_message.Message):
    __slots__ = ("contract_version", "backend_id", "tenant_scope", "job_id", "attempt_id", "fence", "provider", "purpose", "input", "expected_active_audio_revision", "plan", "sample", "artifact_prefix", "manifest_key", "deadline", "heartbeat_seconds", "lease_seconds", "correlation_id")
    CONTRACT_VERSION_FIELD_NUMBER: _ClassVar[int]
    BACKEND_ID_FIELD_NUMBER: _ClassVar[int]
    TENANT_SCOPE_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
    FENCE_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_FIELD_NUMBER: _ClassVar[int]
    PURPOSE_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_ACTIVE_AUDIO_REVISION_FIELD_NUMBER: _ClassVar[int]
    PLAN_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_PREFIX_FIELD_NUMBER: _ClassVar[int]
    MANIFEST_KEY_FIELD_NUMBER: _ClassVar[int]
    DEADLINE_FIELD_NUMBER: _ClassVar[int]
    HEARTBEAT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    LEASE_SECONDS_FIELD_NUMBER: _ClassVar[int]
    CORRELATION_ID_FIELD_NUMBER: _ClassVar[int]
    contract_version: str
    backend_id: str
    tenant_scope: str
    job_id: str
    attempt_id: str
    fence: int
    provider: str
    purpose: str
    input: SourceIdentity
    expected_active_audio_revision: str
    plan: CleanPlan
    sample: SampleInterval
    artifact_prefix: str
    manifest_key: str
    deadline: str
    heartbeat_seconds: int
    lease_seconds: int
    correlation_id: str
    def __init__(self, contract_version: _Optional[str] = ..., backend_id: _Optional[str] = ..., tenant_scope: _Optional[str] = ..., job_id: _Optional[str] = ..., attempt_id: _Optional[str] = ..., fence: _Optional[int] = ..., provider: _Optional[str] = ..., purpose: _Optional[str] = ..., input: _Optional[_Union[SourceIdentity, _Mapping]] = ..., expected_active_audio_revision: _Optional[str] = ..., plan: _Optional[_Union[CleanPlan, _Mapping]] = ..., sample: _Optional[_Union[SampleInterval, _Mapping]] = ..., artifact_prefix: _Optional[str] = ..., manifest_key: _Optional[str] = ..., deadline: _Optional[str] = ..., heartbeat_seconds: _Optional[int] = ..., lease_seconds: _Optional[int] = ..., correlation_id: _Optional[str] = ...) -> None: ...

class StorageGrant(_message.Message):
    __slots__ = ("reference", "token", "expires_at")
    REFERENCE_FIELD_NUMBER: _ClassVar[int]
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    EXPIRES_AT_FIELD_NUMBER: _ClassVar[int]
    reference: str
    token: str
    expires_at: str
    def __init__(self, reference: _Optional[str] = ..., token: _Optional[str] = ..., expires_at: _Optional[str] = ...) -> None: ...

class ExecuteAttemptRequest(_message.Message):
    __slots__ = ("ticket", "source_read_grant", "artifact_write_grant")
    TICKET_FIELD_NUMBER: _ClassVar[int]
    SOURCE_READ_GRANT_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_WRITE_GRANT_FIELD_NUMBER: _ClassVar[int]
    ticket: AttemptTicket
    source_read_grant: StorageGrant
    artifact_write_grant: StorageGrant
    def __init__(self, ticket: _Optional[_Union[AttemptTicket, _Mapping]] = ..., source_read_grant: _Optional[_Union[StorageGrant, _Mapping]] = ..., artifact_write_grant: _Optional[_Union[StorageGrant, _Mapping]] = ...) -> None: ...

class ManifestReference(_message.Message):
    __slots__ = ("backend_id", "tenant_scope", "job_id", "attempt_id", "fence", "object_key", "object_version", "sha256", "size_bytes", "outcome", "error_code")
    BACKEND_ID_FIELD_NUMBER: _ClassVar[int]
    TENANT_SCOPE_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
    FENCE_FIELD_NUMBER: _ClassVar[int]
    OBJECT_KEY_FIELD_NUMBER: _ClassVar[int]
    OBJECT_VERSION_FIELD_NUMBER: _ClassVar[int]
    SHA256_FIELD_NUMBER: _ClassVar[int]
    SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_FIELD_NUMBER: _ClassVar[int]
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    backend_id: str
    tenant_scope: str
    job_id: str
    attempt_id: str
    fence: int
    object_key: str
    object_version: str
    sha256: str
    size_bytes: int
    outcome: str
    error_code: str
    def __init__(self, backend_id: _Optional[str] = ..., tenant_scope: _Optional[str] = ..., job_id: _Optional[str] = ..., attempt_id: _Optional[str] = ..., fence: _Optional[int] = ..., object_key: _Optional[str] = ..., object_version: _Optional[str] = ..., sha256: _Optional[str] = ..., size_bytes: _Optional[int] = ..., outcome: _Optional[str] = ..., error_code: _Optional[str] = ...) -> None: ...
