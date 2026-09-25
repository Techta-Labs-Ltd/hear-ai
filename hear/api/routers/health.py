from fastapi import APIRouter, HTTPException

from hear.health.service import HealthService


class HealthRouter:
    def __init__(self, health: HealthService) -> None:
        self._health = health
        self.router = APIRouter()
        self.router.add_api_route("/healthz", self.healthz, methods=["GET"])
        self.router.add_api_route("/readyz", self.readyz, methods=["GET"])
        self.router.add_api_route("/capabilities", self.capabilities, methods=["GET"])
        self.router.add_api_route("/drain", self.drain, methods=["POST"])

    async def healthz(self) -> dict:
        snapshot = self._health.snapshot()
        return {
            "status": "healthy",
            "role": snapshot.role,
            "active": snapshot.active,
            "capacity": snapshot.capacity,
            "checks": snapshot.checks,
        }

    async def readyz(self) -> dict:
        snapshot = self._health.snapshot()
        if not snapshot.ready:
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "not_ready",
                    "role": snapshot.role,
                    "active": snapshot.active,
                    "capacity": snapshot.capacity,
                    "checks": snapshot.checks,
                },
            )
        return {
            "status": "ready",
            "role": snapshot.role,
            "active": snapshot.active,
            "capacity": snapshot.capacity,
            "checks": snapshot.checks,
        }

    async def capabilities(self) -> dict:
        return self._health.capabilities()

    async def drain(self) -> dict:
        self._health.set_draining(True)
        return {"status": "draining"}
