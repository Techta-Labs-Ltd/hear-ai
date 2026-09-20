from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    RAY_ADDRESS: str = "local"
    RAY_DASHBOARD_HOST: str = "127.0.0.1"
    RAY_DASHBOARD_PORT: int = 8324
    HTTP_HOST: str = "0.0.0.0"
    HTTP_PORT: int = 8000
    GRPC_PORT: int = 50051
    GRPC_APPLICATION_NAME: str = "hear"
    GATEWAY_REPLICA_COUNT: int = 2
    LOG_LEVEL: str = "INFO"
    BACKEND_REGISTRY_JSON: str = ""
    STORAGE_CONTEXT_ENCRYPTION_KEY: str = ""
    WHISPER_BATCH_SIZE: int = 36
    WHISPER_CHUNK_SECONDS: int = 600
    WHISPER_LONG_AUDIO_BATCH_SIZE: int = 4
    WHISPER_VAD_ONSET: float = 0.65
    WHISPER_VAD_OFFSET: float = 0.50
    WHISPER_MIN_AVG_LOGPROB: float = -0.75
    QWEN_ASR_DTYPE: str = "bfloat16"
    QWEN_ASR_DEVICE_MAP: str = "cuda:0"
    QWEN_LLM_ENABLED: bool = False
    ORCHESTRATOR_MAX_CONCURRENT_JOBS: int = 8
    ORCHESTRATOR_MAX_CONCURRENT_JOBS_PER_USER: int = 3
    MAGIC_CLEAN_REPLICA_COUNT: int = 1
    MAGIC_CLEAN_CHUNK_SECONDS: int = 60
    MAGIC_CLEAN_CHUNK_OVERLAP_SECONDS: float = 2.0
    MAGIC_CLEAN_STREAMING_THRESHOLD_SECONDS: int = 300
    MAGIC_CLEAN_ENGINE_REVISION: str = "safe-v1"
    MAGIC_CLEAN_MP3_BITRATE_KBPS: int = 96
    MAGIC_CLEAN_CLEANUP_GRACE_SECONDS: float = 30 * 60
    MAGIC_CLEAN_STORAGE_CREDENTIAL_MIN_TTL_SECONDS: float = 24 * 60 * 60
    FISH_SPEECH_REPLICA_COUNT: int = 1
    GPU_ON_DEMAND_IDLE_SECONDS: float = 15.0
    ORCHESTRATOR_JOB_TYPE_LIMITS: dict[str, int] = {
        "pipeline": 4,
        "transcription": 4,
        "audio_tag": 6,
        "categorization": 6,
        "magic_clean": 1,
        "rebuild": 2,
        "reconstruction": 1,
        "discovery": 4,
    }
    ORCHESTRATOR_RECOVERY_SECONDS: float = 15.0
    DATABASE_URL: str = ""
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_RECYCLE: int = 1800
    DB_POOL_PRE_PING: bool = True
    DB_STATEMENT_TIMEOUT_MS: int = 60000

    PIPELINE_MP3_BITRATE_KBPS: int = 96

    DISCOVERY_METADATA_ENABLED: bool = True
    DISCOVERY_MAX_SEARCH_PHRASES: int = 12
    DISCOVERY_MAX_NEW_TOKENS: int = 1100
    HEAR_TEMP_DIR: str = str(PROJECT_ROOT / "audio")
    AUDIO_CLEANUP_INTERVAL_SECONDS: float = 300.0
    AUDIO_MAX_AGE_SECONDS: float = 24 * 60 * 60
    AUDIO_DOWNLOAD_MAX_BYTES: int = Field(default=4 * 1024**3, gt=0)
    AUDIO_DOWNLOAD_READ_TIMEOUT_SECONDS: float = Field(default=60.0, gt=0, allow_inf_nan=False)
    AUDIO_DECODE_TIMEOUT_SECONDS: float = Field(default=1200.0, gt=0, allow_inf_nan=False)
    MODEL_CACHE_DIR: str = "/workspace/models"
    QWEN_ASR_MODEL_PATH: str = "/workspace/models/qwen3-asr-1.7b"
    ALIGNER_MODEL_PATH: str = "/workspace/models/qwen3-forced-aligner"
    LLM_MODEL_PATH: str = "/workspace/models/qwen2.5-7b-instruct"
    TOXIC_MODEL_PATH: str = "/workspace/models/toxic-bert"
    SENTIMENT_MODEL_PATH: str = "/workspace/models/twitter-roberta-sentiment"
    NLI_MODEL_PATH: str = "/workspace/models/nli-distilroberta"
    MOSSFORMER_MODEL_PATH: str = "/workspace/models/mossformer2-se-48k"
    DEMUCS_MODEL: str = "htdemucs"
    DEMUCS_MODEL_PATH: str = "/workspace/models/demucs"

    FISH_SPEECH_TTS_ENABLED: bool = True
    FISH_SPEECH_HOME: str = "/workspace/fish-speech"
    FISH_SPEECH_CHECKPOINT_PATH: str = "/workspace/models/fish-speech/s2-pro"
    FISH_SPEECH_CODEC_PATH: str = "/workspace/models/fish-speech/s2-pro/codec.pth"
    FISH_SPEECH_BNB_MODE: str = "nf4"
    REGENERATION_PREVIEW_TTL_SECONDS: int = 3600

    MAX_CONCURRENT_EDIT_TRANSCRIPT_JOBS: int = 2
    VOICE_PROFILES_DIR: str = ""
    VOICE_PROFILE_MAX_AGE_HOURS: int = 168
    EDIT_PHRASE_EXPANSION_WORDS: int = 1
    EDIT_MERGE_GAP_SECONDS: float = 1.5
    EDIT_MAX_BATCH_WORDS: int = 80
    EDIT_MAX_BATCH_DURATION: float = 30.0

    SENTRY_DSN: str = ""
    SENTRY_TRACES_SAMPLE_RATE: float = 0.3
    ENVIRONMENT: str = "production"
    ENABLE_DOCS: bool = False

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", PROJECT_ROOT.parent / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
