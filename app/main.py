import os

os.environ["HF_HUB_OFFLINE"] = os.getenv("HF_HUB_OFFLINE", "0")
os.environ["TRANSFORMERS_OFFLINE"] = os.getenv("TRANSFORMERS_OFFLINE", "0")
os.environ["HF_DATASETS_OFFLINE"] = os.getenv("HF_DATASETS_OFFLINE", "0")

import logging

import sentry_sdk
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration

from app.api.router import api_router
from app.config import settings

logger = logging.getLogger(__name__)

OPENAPI_DESCRIPTION = """
## Hear AI – Audio Intelligence Service

Hear AI provides a complete audio processing pipeline including:

- **Pipeline** – transcription, moderation, categorization
- **Magic Clean** – standalone vocal isolation & noise removal via Demucs
- **Transcription** – standalone speech-to-text with WhisperX distil-large-v3
- **Categorization** – standalone topic tagging
- **Moderation** – standalone content safety analysis
- **Reconstruction** – segment regeneration with Fish Speech voice cloning

### Authentication
All endpoints (except `/health`) require a service key via **either**:
- `X-Service-Key` header
- `Authorization: Bearer <key>` header
"""

TAGS_METADATA: list[dict] = [
    {"name": "System", "description": "Health checks and system status"},
    {"name": "Pipeline", "description": "Track-first pipeline"},
    {"name": "Transcription", "description": "Standalone speech-to-text jobs"},
    {"name": "Enhancement", "description": "Standalone audio enhancement"},
    {"name": "Categorization", "description": "Text-based topic categorization"},
    {"name": "Discovery", "description": "Discovery catalog"},
    {"name": "Moderation", "description": "Content safety analysis"},
    {"name": "Realtime", "description": "SSE and WebSocket streaming"},
    {"name": "Jobs", "description": "Job status polling"},
]


def _init_sentry() -> None:
    if not settings.SENTRY_DSN:
        return
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        traces_sample_rate=settings.SENTRY_TRACES_SAMPLE_RATE,
        environment=settings.ENVIRONMENT,
        integrations=[
            FastApiIntegration(transaction_style="endpoint"),
            StarletteIntegration(transaction_style="endpoint"),
        ],
        send_default_pii=False,
    )


def create_app() -> FastAPI:
    _init_sentry()

    app = FastAPI(
        title="Hear AI Service",
        version="3.0.0",
        description=OPENAPI_DESCRIPTION,
        openapi_tags=TAGS_METADATA,
        docs_url="/docs" if settings.ENABLE_DOCS else None,
        redoc_url="/redoc" if settings.ENABLE_DOCS else None,
        contact={"name": "Techta Labs", "url": "https://techta.co"},
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def _global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        sentry_sdk.capture_exception(exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    app.include_router(api_router)

    def _custom_openapi() -> dict:
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            tags=app.openapi_tags,
            routes=app.routes,
        )
        app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = _custom_openapi
    return app


app = create_app()
