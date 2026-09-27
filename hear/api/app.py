from collections.abc import Awaitable, Callable

from fastapi import FastAPI

from hear.api.routers.health import HealthRouter
from hear.api.routers.jobs import JobsRouter
from hear.health.service import RuntimeReadiness
from hear.runtime.pod import PodRuntime


class RuntimeApi:
    def __init__(
        self,
        readiness: RuntimeReadiness,
        *,
        drain: Callable[[], Awaitable[None]] | None = None,
        pod_runtime: PodRuntime | None = None,
        pod_api_key: str = "",
        enable_docs: bool = False,
    ) -> None:
        self._readiness = readiness
        self._app = FastAPI(
            title="Hear AI Runtime",
            version="6.0.0",
            docs_url="/docs" if enable_docs else None,
            redoc_url="/redoc" if enable_docs else None,
            openapi_url="/openapi.json" if enable_docs else None,
        )
        self._app.include_router(HealthRouter(readiness, drain).router)
        if pod_runtime is not None:
            self._app.include_router(JobsRouter(pod_runtime, pod_api_key).router)

    @property
    def app(self) -> FastAPI:
        return self._app
