from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeSettings(BaseSettings):
    worker_role: Literal["pipeline", "transcription", "reconstruction", "magic_clean"] = Field(
        default="transcription",
        validation_alias="HEAR_WORKER_ROLE",
    )
    rabbitmq_url: str = Field(default="", validation_alias="HEAR_RABBITMQ_URL")
    rabbitmq_queue: str = Field(default="", validation_alias="HEAR_RABBITMQ_QUEUE")
    rabbitmq_prefetch: int = Field(default=1, ge=1, le=32, validation_alias="HEAR_RABBITMQ_PREFETCH")
    http_host: str = Field(default="0.0.0.0", validation_alias="HEAR_HTTP_HOST")
    http_port: int = Field(default=8000, ge=1, le=65535, validation_alias="HEAR_HTTP_PORT")
    worker_id: str = Field(default="worker", validation_alias="HEAR_WORKER_ID")
    worker_generation: str = Field(default="1", validation_alias="HEAR_WORKER_GENERATION")
    event_timeout_seconds: float = Field(
        default=15.0,
        gt=0,
        validation_alias="HEAR_EVENT_TIMEOUT_SECONDS",
    )
    backend_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        validation_alias="HEAR_BACKEND_TIMEOUT_SECONDS",
    )

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")
