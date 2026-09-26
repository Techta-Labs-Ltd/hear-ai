import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

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
