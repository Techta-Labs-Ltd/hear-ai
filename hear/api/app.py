from fastapi import FastAPI

from hear.api.routers.health import HealthRouter
from hear.health.service import HealthService


class ApplicationFactory:
    def __init__(self, health: HealthService, enable_docs: bool = False) -> None:
        self._health = health
        self._enable_docs = enable_docs

    def build(self) -> FastAPI:
        app = FastAPI(
            title="Hear AI",
            version="6.0.0",
            docs_url="/docs" if self._enable_docs else None,
            redoc_url="/redoc" if self._enable_docs else None,
            openapi_url="/openapi.json" if self._enable_docs else None,
        )
        app.include_router(HealthRouter(self._health).router)
        return app
