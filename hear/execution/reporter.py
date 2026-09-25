from __future__ import annotations

import httpx

from hear.contracts.events import ExecutionEvent
from hear.contracts.jobs import AttemptEnvelope
from hear.runtime.config import RuntimeSettings


class BackendAttemptClient:
    def __init__(self, settings: RuntimeSettings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(timeout=settings.backend_timeout_seconds)

    async def claim(self, request: AttemptEnvelope) -> str:
        response = await self._client.post(
            f"{str(request.reporting.backend_base_url).rstrip('/')}/internal/ai/attempts/{request.attempt_id}/claim",
            headers={"Authorization": f"Bearer {request.reporting.grant}"},
            json={
                "worker_id": self._settings.worker_id,
                "worker_generation": self._settings.worker_generation,
                "job_type": request.job_type,
            },
        )
        response.raise_for_status()
        body = response.json()
        return str(body.get("decision") or "")

    async def event(self, request: AttemptEnvelope, event: ExecutionEvent) -> None:
        response = await self._client.post(
            f"{str(request.reporting.backend_base_url).rstrip('/')}/internal/ai/attempts/{request.attempt_id}/events",
            headers={"Authorization": f"Bearer {request.reporting.grant}"},
            json=event.model_dump(mode="json"),
        )
        response.raise_for_status()

    async def heartbeat(self, request: AttemptEnvelope) -> None:
        response = await self._client.post(
            f"{str(request.reporting.backend_base_url).rstrip('/')}/internal/ai/attempts/{request.attempt_id}/heartbeat",
            headers={"Authorization": f"Bearer {request.reporting.grant}"},
            json={
                "worker_id": self._settings.worker_id,
                "worker_generation": self._settings.worker_generation,
            },
        )
        response.raise_for_status()

    async def close(self) -> None:
        await self._client.aclose()
