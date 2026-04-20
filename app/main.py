from contextlib import asynccontextmanager

import uvicorn
import sentry_sdk
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration

from app.config import settings
from app.core.category_loader import category_loader
from app.core.keyword_loader import harm_keyword_loader
from app.models.database import init_db
from app.services.registry import (
    transcriber,
    enhancer,
    categorizer,
    moderator,
    synthesizer,
    worker,
)
from app.api.router import api_router


def init_sentry():
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


@asynccontextmanager
async def lifespan(application: FastAPI):
    init_sentry()
    init_db()
    category_loader.load()
    harm_keyword_loader.load()
    print("[STARTUP] Loading ML models...")
    transcriber.load()
    enhancer.load()
    categorizer.load()
    moderator.load()
    synthesizer.load()
    print("[STARTUP] Models loaded. Starting worker...")
    await worker.start()
    print("[STARTUP] Ready.")
    yield
    await worker.stop()


OPENAPI_DESCRIPTION = """
## Hear AI – Audio Intelligence Service

Hear AI provides a complete audio processing pipeline including:

- **Enhancement** – vocal isolation & noise removal via Demucs
- **Transcription** – speech-to-text with Faster-Whisper (large-v3)
- **Categorization** – LLM-powered topic tagging
- **Moderation** – content safety analysis
- **Reconstruction** – accent-aware speech synthesis (Edge-TTS)

### Authentication
All endpoints (except `/health`) require a service key via **either**:
- `X-Service-Key` header
- `Authorization: Bearer <key>` header
"""

TAGS_METADATA = [
    {"name": "System", "description": "Health checks and system status"},
    {"name": "Pipeline", "description": "Full audio processing pipeline (enhance → transcribe → categorize → moderate)"},
    {"name": "Transcription", "description": "Standalone speech-to-text jobs"},
    {"name": "Enhancement", "description": "Standalone audio enhancement / vocal isolation jobs"},
    {"name": "Categorization", "description": "Text-based topic categorization"},
    {"name": "Moderation", "description": "Content safety / moderation analysis"},
    {"name": "Realtime", "description": "SSE and WebSocket streaming endpoints"},
    {"name": "Jobs", "description": "Job status polling"},
]

app = FastAPI(
    title="Hear AI Service",
    version="2.0.0",
    description=OPENAPI_DESCRIPTION,
    lifespan=lifespan,
    openapi_tags=TAGS_METADATA,
    docs_url="/docs" if settings.ENABLE_DOCS else None,
    redoc_url="/redoc" if settings.ENABLE_DOCS else None,
    openapi_url="/openapi.json" if settings.ENABLE_DOCS else None,
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
async def global_exception_handler(request: Request, exc: Exception):
    sentry_sdk.capture_exception(exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )


app.include_router(api_router)


def custom_openapi():
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


app.openapi = custom_openapi

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, workers=1, reload=False)
