from __future__ import annotations

import os
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
from pydantic_settings import BaseSettings, SettingsConfigDict

from hear.runtime.roles import WorkerRole

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGURED_ENV_FILE = os.environ.get("HEAR_ENV_FILE", "").strip()
ENV_FILES = (
    (Path(CONFIGURED_ENV_FILE),)
    if CONFIGURED_ENV_FILE
    else (PROJECT_ROOT / ".env", PROJECT_ROOT.parent / ".env")
)


class Settings(BaseSettings):
    WHISPER_MIN_AVG_LOGPROB: float = -0.75
    DISCOVERY_METADATA_ENABLED: bool = True
    DISCOVERY_MAX_SEARCH_PHRASES: int = 12
    DISCOVERY_MAX_NEW_TOKENS: int = 1100
    FISH_SPEECH_TTS_ENABLED: bool = True
    EDIT_PHRASE_EXPANSION_WORDS: int = 1
    EDIT_MERGE_GAP_SECONDS: float = 1.5
    EDIT_MAX_BATCH_WORDS: int = 80
    EDIT_MAX_BATCH_DURATION: float = 30.0
    DNSMOS_MODEL_PATH: str = "/models/dnsmos/sig_bak_ovr.onnx"
    HEAR_TEMP_DIR: str = "/audio"
    AUDIO_MAX_AGE_SECONDS: float = 24 * 60 * 60

    model_config = SettingsConfigDict(
        env_file=ENV_FILES,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()


class RuntimeSettings(BaseModel):
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
    backend_internal_url: AnyHttpUrl | None = Field(default=None, alias="HEAR_BACKEND_INTERNAL_URL")
    backend_service_key: SecretStr | None = Field(default=None, alias="HEAR_BACKEND_SERVICE_KEY")
    cleaner_certification_path: Path | None = Field(
        default=None, alias="HEAR_CLEANER_CERTIFICATION_PATH"
    )
    cleaner_certification_sha256: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[a-f0-9]{64}$",
        alias="HEAR_CLEANER_CERTIFICATION_SHA256",
    )
    cleaner_lock_dir: Path = Field(
        default=Path("/tmp/hear-cleaner-locks"), alias="HEAR_CLEANER_LOCK_DIR"
    )
    optional_engine_mode: Literal["available", "certified"] = Field(
        default="available", alias="HEAR_OPTIONAL_ENGINE_MODE"
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
    pipeline_mp3_bitrate_kbps: int = Field(default=96, gt=0, alias="PIPELINE_MP3_BITRATE_KBPS")
    whisper_batch_size: int = Field(default=36, gt=0, alias="WHISPER_BATCH_SIZE")
    whisper_chunk_seconds: int = Field(default=600, gt=0, alias="WHISPER_CHUNK_SECONDS")
    whisper_long_audio_batch_size: int = Field(
        default=4, gt=0, alias="WHISPER_LONG_AUDIO_BATCH_SIZE"
    )
    whisper_vad_onset: float = Field(default=0.65, gt=0, lt=1, alias="WHISPER_VAD_ONSET")
    whisper_vad_offset: float = Field(default=0.50, ge=0, lt=1, alias="WHISPER_VAD_OFFSET")
    qwen_asr_dtype: str = Field(default="bfloat16", min_length=1, alias="QWEN_ASR_DTYPE")
    qwen_asr_device_map: str = Field(default="cuda:0", min_length=1, alias="QWEN_ASR_DEVICE_MAP")
    qwen_llm_gpu_memory_utilization: float = Field(
        default=0.75, gt=0, le=1, alias="QWEN_LLM_GPU_MEMORY_UTILIZATION"
    )
    discovery_max_new_tokens: int = Field(default=1100, gt=0, alias="DISCOVERY_MAX_NEW_TOKENS")
    fish_speech_home: Path = Field(default=Path("/fish-speech"), alias="FISH_SPEECH_HOME")
    fish_speech_bnb_mode: Literal["nf4"] = Field(default="nf4", alias="FISH_SPEECH_BNB_MODE")
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
        "cleaner_lock_dir",
        "host_job_lock_path",
        mode="before",
    )
    @classmethod
    def validate_required_paths(cls, value: object) -> object:
        if value is None or not str(value).strip():
            raise ValueError("runtime_path_must_not_be_empty")
        return value

    @field_validator(
        "cleaner_certification_path",
        "sound_cleanup_bundle",
        "sound_cleanup_separator_bundle",
        "fish_speech_model_root",
        mode="before",
    )
    @classmethod
    def normalize_optional_certification_path(cls, value: object) -> object:
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

    def required(self, field_name: str) -> str:
        field = type(self).model_fields[field_name]
        value = getattr(self, field_name)
        rendered = value.get_secret_value() if isinstance(value, SecretStr) else str(value or "")
        if not rendered.strip():
            alias = field.alias or field_name
            raise RuntimeError(f"missing_runtime_setting:{alias}")
        return rendered
