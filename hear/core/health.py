import asyncio
import time
import uuid
from collections.abc import Callable, Mapping

from ray import serve


class RayHealthSnapshot:
    def __init__(self, application: str, model_names: tuple[str, ...]):
        self._application = application
        self._model_names = model_names

    def __call__(self) -> dict:
        application = serve.status().applications.get(self._application)
        if application is None:
            return {"control_ready": False, "capabilities": {}}
        deployments = application.deployments
        capabilities = {}
        for name in self._model_names:
            deployment = deployments.get(name)
            if deployment is None:
                capabilities[name] = {"state": "disabled", "replicas": 0}
                continue
            status = getattr(deployment.status, "value", deployment.status)
            states = {
                getattr(state, "value", state): count
                for state, count in deployment.replica_states.items()
            }
            running = states.get("RUNNING", 0)
            if status in {"UNHEALTHY", "DEPLOY_FAILED"}:
                state = "unavailable"
            elif running and status == "HEALTHY":
                state = "ready"
            elif sum(states.values()) == 0 and status == "HEALTHY":
                state = "cold"
            else:
                state = "warming"
            capabilities[name] = {"state": state, "replicas": running}
        gateway = deployments.get("grpc_gateway")
        control_ready = gateway is not None and any(
            getattr(state, "value", state) == "RUNNING" and count > 0
            for state, count in gateway.replica_states.items()
        )
        return {"control_ready": control_ready, "capabilities": capabilities}


class ServiceHealth:
    def __init__(
        self,
        read_snapshot: Callable[[], Mapping],
        timeout_seconds: float = 2.0,
        cache_seconds: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        if timeout_seconds <= 0 or cache_seconds < 0:
            raise ValueError("invalid_health_timeouts")
        self._read_snapshot = read_snapshot
        self._timeout = timeout_seconds
        self._cache_seconds = cache_seconds
        self._clock = clock
        self._epoch = str(uuid.uuid4())
        self._snapshot: dict | None = None
        self._updated_at = float("-inf")
        self._refresh_task: asyncio.Task[Mapping] | None = None

    async def read(self, queue: dict) -> dict:
        now = self._clock()
        error = None
        if self._snapshot is None or now - self._updated_at >= self._cache_seconds:
            if self._refresh_task is None:
                self._refresh_task = asyncio.create_task(asyncio.to_thread(self._read_snapshot))
            try:
                self._snapshot = dict(
                    await asyncio.wait_for(
                        asyncio.shield(self._refresh_task), timeout=self._timeout
                    )
                )
                self._updated_at = self._clock()
                self._refresh_task = None
            except TimeoutError:
                error = "health_probe_timeout"
            except Exception:
                error = "health_probe_failed"
                self._refresh_task = None
        snapshot = self._snapshot or {}
        capabilities = snapshot.get("capabilities", {}) if error is None else {}
        control_ready = bool(snapshot.get("control_ready")) and error is None
        degraded = any(value["state"] == "unavailable" for value in capabilities.values())
        return {
            "status": "unavailable" if not control_ready else "degraded" if degraded else "healthy",
            "control_ready": control_ready,
            "service_epoch": self._epoch,
            "capabilities": capabilities,
            "models_loaded": [
                name for name, value in capabilities.items() if value["state"] == "ready"
            ],
            "gpu_metrics_available": False,
            "active_jobs": queue.get("active", 0),
            "queued_jobs": queue.get("queued", 0),
            "error": error,
        }
