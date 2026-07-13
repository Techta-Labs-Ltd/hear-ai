from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    AI_SERVICE_SECRET: str = "change-me"
    HEAR_BACKEND_URL: str = "http://localhost:3000"
    WHISPER_BATCH_SIZE: int = 36
    QWEN_LLM_ENABLED: bool = False
    JOB_MAX_RETRIES: int = 3
    DATABASE_URL: str = ""
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_RECYCLE: int = 1800
    DB_POOL_PRE_PING: bool = True
    DB_STATEMENT_TIMEOUT_MS: int = 60000

    B2_KEY_ID: str = ""
    B2_APPLICATION_KEY: str = ""
    B2_BUCKET_NAME: str = "hear-dev-uploads"
    B2_ENDPOINT_URL: str = "https://s3.eu-central-003.backblazeb2.com"
    B2_ENHANCED_PREFIX: str = "enhanced/"
    B2_PIPELINE_MP3_PREFIX: str = "pipeline-source-mp3/"
    PIPELINE_SPEED_MULTIPLIERS: str = "0.5,0.75,0.9,1.1,1.25,1.5,2.0,3.0"
    PIPELINE_MP3_BITRATE_KBPS: int = 96

    CATEGORIES_FILE: str = "./data/categories.txt"
    DISCOVERY_TAXONOMY_FILE: str = "./data/discovery_taxonomy.txt"
    CATEGORIZER_SYNC_TAXONOMY: bool = True
    DISCOVERY_METADATA_ENABLED: bool = True
    DISCOVERY_MAX_SEARCH_PHRASES: int = 12
    DISCOVERY_MAX_NEW_TOKENS: int = 1100
    HARM_KEYWORDS_FILE: str = "./data/harm_keywords.txt"
    DEMUCS_MODEL: str = "htdemucs"
    MODEL_CACHE_DIR: str = "/opt/ml/models"

    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = "gpt-4o-mini"
    MODERATION_AUTO_LEARN: bool = False
    FISH_SPEECH_TTS_ENABLED: bool = True
    FISH_SPEECH_TTS_SERVER_URL: str = "http://localhost:8080"
    FISH_SPEECH_HOME: str = "/root/fish-speech"
    FISH_SPEECH_CHECKPOINT_PATH: str = "checkpoints/s2-pro"
    FISH_SPEECH_CODEC_PATH: str = "checkpoints/s2-pro/codec.pth"
    REGENERATION_PREVIEW_TTL_SECONDS: int = 3600

    MAX_CONCURRENT_EDIT_TRANSCRIPT_JOBS: int = 1
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
        env_file=".env",
        extra="ignore",
    )

settings = Settings()
