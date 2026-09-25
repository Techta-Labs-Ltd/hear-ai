from __future__ import annotations

import httpx

from hear.contracts.events import ExecutionEvent
from hear.contracts.jobs import AttemptEnvelope, ClaimDecision, WorkerIdentity
from hear.contracts.outcomes import ExecutionOutcome


class BackendAttemptClient:
    def __init__(
        self,
        worker: WorkerIdentity,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._worker = worker
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(20.0))
        self._owns_client = client is None

    @staticmethod
    def _url(envelope: AttemptEnvelope, suffix: str) -> str:
        base = str(envelope.backend_base_url).rstrip("/")
        return f"{base}/internal/ai/attempts/{envelope.attempt_id}/{suffix}"

    @staticmethod
    def _headers(envelope: AttemptEnvelope) -> dict[str, str]:
        return {"X-AI-Attempt-Grant": envelope.reporting_grant}

    async def claim(self, envelope: AttemptEnvelope) -> ClaimDecision:
        response = await self._client.post(
            self._url(envelope, "claim"),
            headers=self._headers(envelope),
            json=self._worker.model_dump(mode="json"),
        )
        response.raise_for_status()
        return ClaimDecision(response.json()["decision"])

    async def heartbeat(self, envelope: AttemptEnvelope, sequence: int) -> None:
        response = await self._client.post(
            self._url(envelope, "heartbeat"),
            headers=self._headers(envelope),
            json={
                "worker_id": self._worker.worker_id,
                "generation": self._worker.generation,
                "sequence": sequence,
            },
        )
        response.raise_for_status()

    async def event(self, envelope: AttemptEnvelope, event: ExecutionEvent) -> None:
        response = await self._client.post(
            self._url(envelope, "events"),
            headers=self._headers(envelope),
            json=event.model_dump(mode="json"),
        )
        response.raise_for_status()

    async def outcome(self, envelope: AttemptEnvelope, outcome: ExecutionOutcome) -> None:
        response = await self._client.post(
            self._url(envelope, "outcome"),
            headers=self._headers(envelope),
            json=outcome.model_dump(mode="json"),
        )
        response.raise_for_status()

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()