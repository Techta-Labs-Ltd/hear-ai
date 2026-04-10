from contextlib import asynccontextmanager

import sentry_sdk
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration

from app.config import settings
from app.core.category_loader import category_loader
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


app = FastAPI(
    title="Hear AI Service",
    version="2.0.0",
    lifespan=lifespan,
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, workers=1, reload=False)
