from urllib.parse import quote_plus

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    AI_SERVICE_SECRET: str = "change-me"
    HEAR_BACKEND_URL: str = "http://localhost:3000"
    HEAR_CALLBACK_URL: str = ""
    WHISPER_MODEL_SIZE: str = "distil-large-v3"
    WHISPER_BEAM_SIZE: int = 1
    WHISPER_WORD_TIMESTAMPS: bool = False
    QWEN_LLM_ENABLED: bool = False
    MAX_CONCURRENT_GPU_JOBS: int = 1
    MAX_CONCURRENT_JOBS: int = 1
    MAX_CONCURRENT_PIPELINE_JOBS: int = 1
    MAX_CONCURRENT_MAGIC_CLEAN_JOBS: int = 1
    JOB_MAX_RETRIES: int = 8
    CALLBACK_RETRY_POLL_SECONDS: int = 45
    DATABASE_URL: str = ""
    POSTGRES_USER: str = "hear_user"
    POSTGRES_PASSWORD: str = ""
    POSTGRES_DB: str = "hear_ai"
    POSTGRES_HOST: str = "127.0.0.1"
    POSTGRES_PORT: int = 5432
    POSTGRES_SSLMODE: str = "disable"
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_RECYCLE: int = 1800
    DB_POOL_PRE_PING: bool = True
    DB_STATEMENT_TIMEOUT_MS: int = 60000
    HEAR_TMP_DIR: str = ""
    HEAR_TEMP_RETENTION_SECONDS: int = 172800
    HEAR_TEMP_SWEEP_INTERVAL_SECONDS: int = 3600

    B2_KEY_ID: str = ""
    B2_APPLICATION_KEY: str = ""
    B2_BUCKET_NAME: str = "hear-audio-assets"
    B2_ENDPOINT_URL: str = "https://s3.eu-central-003.backblazeb2.com"
    B2_ENHANCED_PREFIX: str = "enhanced/"
    B2_PIPELINE_MP3_PREFIX: str = "pipeline-source-mp3/"
    PIPELINE_SPEED_MULTIPLIERS: str = "0.5,0.75,0.9,1.1,1.25,1.5,2.0,3.0"
    PIPELINE_MP3_BITRATE_KBPS: int = 96

    CATEGORIES_FILE: str = "./data/categories.txt"
    DISCOVERY_TAXONOMY_FILE: str = "./data/discovery_taxonomy.txt"
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
    HIGGS_AUDIO_ENABLED: bool = False
    HIGGS_AUDIO_MODULE: str = "boson_multimodal"
    HIGGS_AUDIO_REPO_DIR: str = "/workspace/higgs-audio"
    HIGGS_AUDIO_VOICE: str = "en_us_001"
    HIGGS_AUDIO_MODEL_PATH: str = "bosonai/higgs-audio-v2-generation-3B-base"
    HIGGS_AUDIO_TOKENIZER_PATH: str = "bosonai/higgs-audio-v2-tokenizer"
    HIGGS_AUDIO_SYSTEM_PROMPT: str = "Generate audio following instruction."

    SENTRY_DSN: str = ""
    SENTRY_TRACES_SAMPLE_RATE: float = 0.3
    ENVIRONMENT: str = "production"
    ENABLE_DOCS: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    @model_validator(mode="after")
    def build_database_url_from_postgres(self) -> "Settings":
        if (self.DATABASE_URL or "").strip():
            return self
        user = quote_plus(self.POSTGRES_USER)
        password = quote_plus(self.POSTGRES_PASSWORD or "")
        host = self.POSTGRES_HOST
        port = self.POSTGRES_PORT
        db = self.POSTGRES_DB
        ssl = self.POSTGRES_SSLMODE
        self.DATABASE_URL = (
            f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}?sslmode={ssl}"
        )
        return self


settings = Settings()
