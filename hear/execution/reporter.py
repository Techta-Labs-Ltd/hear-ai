from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx

from hear.contracts.events import ExecutionEvent
from hear.contracts.jobs import AttemptClaim, AttemptEnvelope, WorkerIdentity
from hear.contracts.outcomes import ExecutionOutcome
from hear.contracts.scope import ExecutionScope
from hear.runtime.ownership import BackendRegistry, DeploymentOwnership


class BackendAttemptClient:
    def __init__(
        self,
        worker: WorkerIdentity,
        backend_internal_url: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        normalized_base = backend_internal_url.strip().rstrip("/")
        if not normalized_base:
            raise ValueError("backend_internal_url_required")
        self._ownership = DeploymentOwnership.load()
        if self._ownership is not None:
            self._ownership.require_reporting_origin(normalized_base)
        self._worker = worker
        self._backend_internal_url = normalized_base
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(20.0))
        self._owns_client = client is None

    def _url(self, envelope: AttemptEnvelope, suffix: str) -> str:
        attempt_id = quote(envelope.attempt_id, safe="")
        base = (
            self._ownership.callback_url(envelope)
            if isinstance(self._ownership, BackendRegistry)
            else self._backend_internal_url
        )
        return f"{base}/internal/ai/attempts/{attempt_id}/{suffix}"

    def _headers(self, envelope: AttemptEnvelope) -> dict[str, str]:
        return {
            "X-AI-Attempt-Grant": envelope.reporting_grant,
            "X-AI-Worker-ID": self._worker.worker_id,
            "X-AI-Worker-Generation": self._worker.generation,
        }

    async def _post(
        self,
        envelope: AttemptEnvelope,
        suffix: str,
        payload: dict[str, Any],
    ) -> httpx.Response:
        if self._ownership is not None:
            self._ownership.validate(envelope)
        response = await self._client.post(
            self._url(envelope, suffix),
            headers=self._headers(envelope),
            json=payload,
            follow_redirects=False,
        )
        response.raise_for_status()
        return response

    async def claim(self, envelope: AttemptEnvelope) -> AttemptClaim:
        response = await self._post(
            envelope,
            "claim",
            {
                **self._worker.model_dump(mode="json"),
                "request_scope_sha256": ExecutionScope.digest(envelope.model_dump(mode="json")),
            },
        )
        return AttemptClaim.model_validate(response.json())

    async def heartbeat(self, envelope: AttemptEnvelope, sequence: int) -> None:
        await self._post(
            envelope,
            "heartbeat",
            {
                "worker_id": self._worker.worker_id,
                "generation": self._worker.generation,
                "sequence": sequence,
            },
        )

    async def event(self, envelope: AttemptEnvelope, event: ExecutionEvent) -> None:
        await self._post(envelope, "events", event.model_dump(mode="json"))

    async def outcome(self, envelope: AttemptEnvelope, outcome: ExecutionOutcome) -> None:
        if (
            outcome.job_id,
            outcome.attempt_id,
            outcome.track_id,
            outcome.source_revision,
            outcome.job_type,
        ) != (
            envelope.job_id,
            envelope.attempt_id,
            envelope.track_id,
            envelope.source.revision,
            envelope.job_type,
        ):
            raise ValueError("outcome_identity_mismatch")
        await self._post(envelope, "outcome", outcome.model_dump(mode="json"))

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
