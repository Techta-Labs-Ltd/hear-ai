from fastapi import FastAPI

from hear.api.routers.health import HealthRouter
from hear.health.service import RuntimeReadiness


class RuntimeApi:
    def __init__(self, readiness: RuntimeReadiness, *, enable_docs: bool = False) -> None:
        self._readiness = readiness
        self._app = FastAPI(
            title="Hear AI Runtime",
            version="6.0.0",
            docs_url="/docs" if enable_docs else None,
            redoc_url="/redoc" if enable_docs else None,
            openapi_url="/openapi.json" if enable_docs else None,
        )
        self._app.include_router(HealthRouter(readiness).router)

    @property
    def app(self) -> FastAPI:
        return self._app