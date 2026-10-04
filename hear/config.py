from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

from hear.runtime.roles import WorkerRole

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# RunPod network volumes are FUSE mounts: slow, shared, and not where weights belong.
# Models live on the container's root disk (/models), baked into images for Serverless.
NETWORK_VOLUME_ROOTS = (Path("/workspace"), Path("/runpod-volume"))


class RuntimeSettings(BaseModel):
    """Process configuration read once from the environment; never from files at import time.

    Launchers load an external env file before starting a worker. Model storage
    is always outside the source checkout and every weight directory can be
    redirected per logical model name through ``HEAR_MODEL_PATHS_JSON``.
    """

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    worker_role: WorkerRole = Field(default=WorkerRole.TRANSCRIPTION, alias="HEAR_WORKER_ROLE")
    worker_id: str | None = Field(default=None, alias="HEAR_WORKER_ID")
    worker_generation: str | None = Field(default=None, alias="HEAR_WORKER_GENERATION")
    image_revision: str | None = Field(default=None, alias="HEAR_IMAGE_REVISION")
    engine_revision: str | None = Field(default=None, alias="HEAR_ENGINE_REVISION")
    model_root: Path = Field(default=Path("/models"), alias="HEAR_MODEL_ROOT")
    model_paths: dict[str, Path] = Field(default_factory=dict, alias="HEAR_MODEL_PATHS_JSON")
    magic_clean_model_dir: Path | None = Field(default=None, alias="HEAR_MAGIC_CLEAN_MODEL_DIR")
    model_features: frozenset[str] = Field(default_factory=frozenset, alias="HEAR_MODEL_FEATURES")
    temp_dir: Path = Field(default=Path("/audio"), alias="HEAR_TEMP_DIR")
    min_free_scratch_bytes: int = Field(default=1024**3, ge=0, alias="HEAR_MIN_FREE_SCRATCH_BYTES")
    pod_api_key: SecretStr | None = Field(default=None, alias="HEAR_POD_API_KEY")
    rabbitmq_url: str | None = Field(default=None, alias="HEAR_RABBITMQ_URL")
    pod_max_concurrent_jobs: int = Field(
        default=1,
        ge=1,
        le=64,
        alias="HEAR_POD_MAX_CONCURRENT_JOBS",
    )
    serverless_max_concurrent_jobs: int = Field(
        default=1,
        ge=1,
        le=64,
        alias="HEAR_SERVERLESS_MAX_CONCURRENT_JOBS",
    )
    serverless_preload_models: bool = Field(default=False, alias="HEAR_SERVERLESS_PRELOAD_MODELS")
    backend_internal_url: AnyHttpUrl | None = Field(default=None, alias="HEAR_BACKEND_INTERNAL_URL")
    backend_service_key: SecretStr | None = Field(default=None, alias="HEAR_BACKEND_SERVICE_KEY")
    host_max_concurrent_jobs: int = Field(
        default=1, ge=1, le=16, alias="HEAR_HOST_MAX_CONCURRENT_JOBS"
    )
    host_job_lock_path: Path = Field(
        default=Path("/tmp/hear-ai/host-job.lock"), alias="HEAR_HOST_JOB_LOCK_PATH"
    )
    audio_download_max_bytes: int = Field(
        default=4 * 1024**3, gt=0, alias="AUDIO_DOWNLOAD_MAX_BYTES"
    )
    audio_download_read_timeout_seconds: float = Field(
        default=60.0, gt=0, alias="AUDIO_DOWNLOAD_READ_TIMEOUT_SECONDS"
    )
    audio_decode_timeout_seconds: float = Field(
        default=1200.0, gt=0, alias="AUDIO_DECODE_TIMEOUT_SECONDS"
    )
    whisper_batch_size: int = Field(default=36, gt=0, alias="WHISPER_BATCH_SIZE")
    whisper_chunk_seconds: int = Field(default=600, gt=0, alias="WHISPER_CHUNK_SECONDS")
    whisper_long_audio_batch_size: int = Field(
        default=4, gt=0, alias="WHISPER_LONG_AUDIO_BATCH_SIZE"
    )
    whisper_vad_onset: float = Field(default=0.65, gt=0, lt=1, alias="WHISPER_VAD_ONSET")
    whisper_vad_offset: float = Field(default=0.50, ge=0, lt=1, alias="WHISPER_VAD_OFFSET")
    whisper_min_avg_logprob: float = Field(default=-0.75, alias="WHISPER_MIN_AVG_LOGPROB")
    qwen_asr_dtype: str = Field(default="bfloat16", min_length=1, alias="QWEN_ASR_DTYPE")
    qwen_asr_device_map: str = Field(default="cuda:0", min_length=1, alias="QWEN_ASR_DEVICE_MAP")
    # Absolute vLLM budget (weights + KV cache + activations), so the same value works on
    # any card. The 4-bit Qwen needs about 5.2 GiB of weights; 8.5 GiB leaves ~2 GiB of KV.
    qwen_llm_gpu_memory_gib: float = Field(default=8.5, gt=0, alias="QWEN_LLM_GPU_MEMORY_GIB")
    gpu_idle_eviction_enabled: bool = Field(default=True, alias="HEAR_GPU_IDLE_EVICTION_ENABLED")
    pipeline_idle_ttl_seconds: float = Field(
        default=600, ge=1, le=86400, alias="HEAR_PIPELINE_IDLE_TTL_SECONDS"
    )
    magic_clean_idle_ttl_seconds: float = Field(
        default=300, ge=1, le=86400, alias="HEAR_MAGIC_CLEAN_IDLE_TTL_SECONDS"
    )
    reconstruction_idle_ttl_seconds: float = Field(
        default=1200, ge=1, le=86400, alias="HEAR_RECONSTRUCTION_IDLE_TTL_SECONDS"
    )
    audiosep_idle_ttl_seconds: float = Field(
        default=90, ge=1, le=86400, alias="HEAR_AUDIOSEP_IDLE_TTL_SECONDS"
    )
    discovery_metadata_enabled: bool = Field(default=True, alias="DISCOVERY_METADATA_ENABLED")
    discovery_max_search_phrases: int = Field(
        default=12, ge=1, le=64, alias="DISCOVERY_MAX_SEARCH_PHRASES"
    )
    discovery_max_new_tokens: int = Field(default=1100, gt=0, alias="DISCOVERY_MAX_NEW_TOKENS")
    fish_speech_home: Path = Field(default=Path("/opt/fish-speech"), alias="FISH_SPEECH_HOME")
    # Explicit operator acknowledgement that permission-required model licences are
    # settled. Readiness keeps reporting the blockers; this flag lets the worker serve.
    fish_license_approved: bool = Field(default=False, alias="HEAR_FISH_LICENSE_APPROVED")
    fish_speech_model_root: Path | None = Field(default=None, alias="FISH_SPEECH_MODEL_ROOT")
    magic_clean_scratch_bytes: int = Field(
        default=8 * 1024**3, gt=0, alias="MAGIC_CLEAN_SCRATCH_BYTES"
    )
    magic_clean_max_input_bytes: int = Field(
        default=4 * 1024**3, gt=0, alias="MAGIC_CLEAN_MAX_INPUT_BYTES"
    )
    magic_clean_max_frames: int = Field(default=96000 * 7200, gt=0, alias="MAGIC_CLEAN_MAX_FRAMES")
    sound_cleanup_separator_bundle: Path | None = Field(
        default=None, alias="HEAR_SOUND_CLEANUP_SEPARATOR_BUNDLE"
    )
    sound_cleanup_separator_sha256: str | None = Field(
        default=None, alias="HEAR_SOUND_CLEANUP_SEPARATOR_SHA256", pattern=r"^[a-f0-9]{64}$"
    )
    sound_cleanup_bundle: Path | None = Field(default=None, alias="HEAR_SOUND_CLEANUP_BUNDLE")
    sound_cleanup_bundle_sha256: str | None = Field(
        default=None, alias="HEAR_SOUND_CLEANUP_BUNDLE_SHA256", pattern=r"^[a-f0-9]{64}$"
    )
    magic_clean_model_device: Literal["cpu", "cuda:0"] = Field(
        default="cuda:0", alias="HEAR_MAGIC_CLEAN_MODEL_DEVICE"
    )
    http_host: str = Field(default="0.0.0.0", min_length=1, alias="HTTP_HOST")
    http_port: int = Field(default=8000, ge=1, le=65535, alias="HTTP_PORT")
    enable_docs: bool = Field(default=False, alias="HEAR_ENABLE_DOCS")
    log_level: str = Field(default="info", min_length=1, alias="LOG_LEVEL")

    @field_validator(
        "model_root",
        "fish_speech_model_root",
        "fish_speech_home",
        "magic_clean_model_dir",
        "sound_cleanup_bundle",
        "sound_cleanup_separator_bundle",
        mode="after",
    )
    @classmethod
    def canonical_model_storage(cls, value: Path | None) -> Path | None:
        # Engines open pinned assets with O_NOFOLLOW and reject symlinked paths, so a
        # model root that is itself a symlink (for example /models -> /workspace/...)
        # must be canonicalized once here rather than failing readiness per asset.
        if value is None:
            return None
        resolved = value.expanduser().resolve()
        cls._require_root_storage(resolved)
        return resolved

    @staticmethod
    def _require_root_storage(resolved: Path) -> None:
        if resolved.is_relative_to(PROJECT_ROOT.resolve()):
            raise ValueError("model_storage_must_not_use_source_checkout")
        if any(resolved.is_relative_to(volume) for volume in NETWORK_VOLUME_ROOTS):
            raise ValueError("model_storage_must_not_use_network_volume")

    @field_validator("model_paths", mode="before")
    @classmethod
    def parse_model_paths(cls, value: object) -> dict[str, Path]:
        if value is None or value == "":
            return {}
        raw = json.loads(value) if isinstance(value, str) else value
        if not isinstance(raw, dict) or any(
            not isinstance(name, str) or not name.strip() or not isinstance(path, (str, Path))
            for name, path in raw.items()
        ):
            raise ValueError("invalid_model_paths")
        paths = {}
        for name, path in raw.items():
            candidate = Path(str(path)).expanduser()
            if not candidate.is_absolute():
                raise ValueError(f"model_path_must_be_absolute:{name}")
            resolved = candidate.resolve()
            cls._require_root_storage(resolved)
            paths[name] = resolved
        return paths

    @field_validator("model_features", mode="before")
    @classmethod
    def parse_model_features(cls, value: object) -> frozenset[str]:
        if value is None or value == "":
            return frozenset()
        if isinstance(value, str):
            return frozenset(item.strip() for item in value.split(",") if item.strip())
        if isinstance(value, (list, tuple, set, frozenset)):
            return frozenset(str(item).strip() for item in value if str(item).strip())
        raise ValueError("invalid_model_features")

    @field_validator(
        "model_root",
        "temp_dir",
        "fish_speech_home",
        "host_job_lock_path",
        mode="before",
    )
    @classmethod
    def validate_required_paths(cls, value: object) -> object:
        if value is None or not str(value).strip():
            raise ValueError("runtime_path_must_not_be_empty")
        return value

    @field_validator(
        "sound_cleanup_bundle",
        "sound_cleanup_separator_bundle",
        "fish_speech_model_root",
        "magic_clean_model_dir",
        mode="before",
    )
    @classmethod
    def normalize_optional_path(cls, value: object) -> object:
        if value is None or not str(value).strip():
            return None
        return value

    @model_validator(mode="after")
    def validate_features(self):
        unknown = self.model_features - {"qwen_llm"}
        if unknown:
            raise ValueError("unsupported_model_features:" + ",".join(sorted(unknown)))
        if self.model_features and self.worker_role != WorkerRole.PIPELINE:
            raise ValueError("model_features_require_pipeline_role")
        return self

    @classmethod
    def from_environment(cls, environment: dict[str, str]) -> RuntimeSettings:
        values = dict(environment)
        if not values.get("HEAR_WORKER_ID", "").strip():
            identity = (
                values.get("RUNPOD_POD_ID", "").strip()
                or values.get("RUNPOD_POD_HOSTNAME", "").strip()
                or values.get("HOSTNAME", "").strip()
                or "local"
            )
            role = values.get("HEAR_WORKER_ROLE", WorkerRole.TRANSCRIPTION.value).strip()
            values["HEAR_WORKER_ID"] = f"runpod-{identity}-{role}-01"
        return cls.model_validate(values)

    @property
    def magic_clean_model_directory(self) -> Path:
        return self.magic_clean_model_dir or self.model_root / "magic-clean" / "DeepFilterNet3"

    def required(self, field_name: str) -> str:
        field = type(self).model_fields[field_name]
        value = getattr(self, field_name)
        rendered = value.get_secret_value() if isinstance(value, SecretStr) else str(value or "")
        if not rendered.strip():
            alias = field.alias or field_name
            raise RuntimeError(f"missing_runtime_setting:{alias}")
        return rendered
