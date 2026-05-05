from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    AI_SERVICE_SECRET: str = "change-me"
    HEAR_BACKEND_URL: str = "http://localhost:3000"
    HEAR_CALLBACK_URL: str = ""
    WHISPER_MODEL_SIZE: str = "large-v3"
    MAX_CONCURRENT_GPU_JOBS: int = 2
    MAX_CONCURRENT_JOBS: int = 8
    MAX_CONCURRENT_PIPELINE_JOBS: int = 1
    MAX_CONCURRENT_MAGIC_CLEAN_JOBS: int = 3
    SQLITE_DB_PATH: str = "./data/jobs.db"

    B2_KEY_ID: str = ""
    B2_APPLICATION_KEY: str = ""
    B2_BUCKET_NAME: str = "hear-audio-assets"
    B2_ENDPOINT_URL: str = "https://s3.eu-central-003.backblazeb2.com"
    B2_ENHANCED_PREFIX: str = "enhanced/"

    CATEGORIES_FILE: str = "./data/categories.txt"
    HARM_KEYWORDS_FILE: str = "./data/harm_keywords.txt"
    DEMUCS_MODEL: str = "htdemucs"
    MODEL_CACHE_DIR: str = "/opt/ml/models"

    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = "gpt-4o-mini"
    MODERATION_AUTO_LEARN: bool = False
    HIGGS_AUDIO_ENABLED: bool = True
    HIGGS_AUDIO_VOICE: str = "en_us_001"

    SENTRY_DSN: str = ""
    SENTRY_TRACES_SAMPLE_RATE: float = 0.3
    ENVIRONMENT: str = "production"
    ENABLE_DOCS: bool = True

    class Config:
        env_file = ".env"


settings = Settings()
