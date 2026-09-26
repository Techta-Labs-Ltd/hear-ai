from collections.abc import Awaitable, Callable

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from hear.health.service import RuntimeReadiness


class HealthRouter:
    def __init__(
        self,
        readiness: RuntimeReadiness,
        drain: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._readiness = readiness
        self._drain = drain
        self.router = APIRouter(tags=["system"])
        self.router.add_api_route("/healthz", self.healthz, methods=["GET"])
        self.router.add_api_route("/readyz", self.readyz, methods=["GET"])
        self.router.add_api_route("/capabilities", self.capabilities, methods=["GET"])
        if drain is not None:
            self.router.add_api_route("/drain", self.drain, methods=["POST"])

    async def healthz(self) -> dict:
        return {"status": "healthy"}

    async def readyz(self):
        payload = self._readiness.snapshot()
        return JSONResponse(payload, status_code=200 if payload["status"] == "ready" else 503)

    async def capabilities(self) -> dict:
        return self._readiness.snapshot()
    async def drain(self) -> dict:
        if self._drain is None:
            return {"status": "unsupported"}
        await self._drain()
        return {"status": "draining"}
