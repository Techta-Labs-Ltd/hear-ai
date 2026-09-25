from fastapi import APIRouter
from fastapi.responses import JSONResponse

from hear.health.service import RuntimeReadiness


class HealthRouter:
    def __init__(self, readiness: RuntimeReadiness) -> None:
        self._readiness = readiness
        self.router = APIRouter(tags=["system"])
        self.router.add_api_route("/healthz", self.healthz, methods=["GET"])
        self.router.add_api_route("/readyz", self.readyz, methods=["GET"])
        self.router.add_api_route("/capabilities", self.capabilities, methods=["GET"])

    async def healthz(self) -> dict:
        return {"status": "healthy"}

    async def readyz(self):
        payload = self._readiness.snapshot()
        return JSONResponse(payload, status_code=200 if payload["status"] == "ready" else 503)

    async def capabilities(self) -> dict:
        return self._readiness.snapshot()