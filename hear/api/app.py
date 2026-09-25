from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, HTTPException

from hear.runtime.config import RuntimeSettings
from hear.runtime.pod import PodRuntime


class OperationalApi:
    def __init__(self, settings: RuntimeSettings, runtime: PodRuntime) -> None:
        self._settings = settings
        self._runtime = runtime
        self._router = APIRouter()
        self._router.add_api_route("/healthz", self.health, methods=["GET"])
        self._router.add_api_route("/readyz", self.ready, methods=["GET"])
        self._router.add_api_route("/capabilities", self.capabilities, methods=["GET"])
        self._router.add_api_route("/drain", self.drain, methods=["POST"])

    def build(self) -> FastAPI:
        app = FastAPI(
            title="Hear AI Worker",
            docs_url=None,
            redoc_url=None,
            lifespan=self._lifespan,
        )
        app.include_router(self._router)
        return app

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        del app
        await self._runtime.start()
        try:
            yield
        finally:
            await self._runtime.close()

    async def health(self) -> dict:
        return {
            "status": "healthy",
            "role": self._settings.worker_role,
            "active": self._runtime.active,
        }

    async def ready(self) -> dict:
        if not self._runtime.ready:
            raise HTTPException(status_code=503, detail="worker_not_ready")
        return {
            "status": "ready",
            "role": self._settings.worker_role,
            "active": self._runtime.active,
        }

    async def capabilities(self) -> dict:
        return {
            "role": self._settings.worker_role,
            "jobs": [self._settings.worker_role],
            "active": self._runtime.active,
            "ready": self._runtime.ready,
        }

    async def drain(self) -> dict:
        await self._runtime.drain()
        return {"status": "drained"}
