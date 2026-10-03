"""Transport-agnostic attempt dispatch for the owning backend.

Both transports accept the same ``AttemptEnvelope`` and return the same
``DispatchReceipt``. Execution events and the final ``ExecutionOutcome`` are
always delivered through the worker's backend callbacks, never through the
transport response, so the backend's job state is identical for Pod and
Serverless deployments. The optional RunPod job status is operational insight
only and must not be used as the source of truth for job completion.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Protocol
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, Field

from hear.contracts.jobs import AttemptEnvelope
from hear.queue.topology import RabbitMQTopology
from hear.runtime.roles import WorkerRole


class DispatchTransport(StrEnum):
    POD = "pod"
    SERVERLESS = "serverless"


class DispatchError(RuntimeError):
    def __init__(self, code: str, *, status_code: int | None = None, retryable: bool) -> None:
        super().__init__(code)
        self.code = code
        self.status_code = status_code
        self.retryable = retryable


class DispatchReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    transport: DispatchTransport
    job_id: str = Field(min_length=1, max_length=128)
    run_id: str = Field(min_length=1, max_length=128)
    attempt_id: str = Field(min_length=1, max_length=128)
    track_id: str = Field(min_length=1, max_length=128)
    source_revision: int = Field(ge=1)
    status: Literal["accepted"] = "accepted"
    provider_job_id: str | None = Field(default=None, max_length=256)
    result_delivery: Literal["owning_backend_callback"] = "owning_backend_callback"
    duplicate_policy: Literal["backend_claim_is_authoritative"] = "backend_claim_is_authoritative"


class TransportHealth(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    transport: DispatchTransport
    ready: bool
    detail: dict[str, Any] = Field(default_factory=dict)


class ServerlessJobStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_job_id: str
    state: Literal[
        "IN_QUEUE", "IN_PROGRESS", "COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT", "UNKNOWN"
    ]
    events: tuple[dict[str, Any], ...] = ()
    error: str | None = None


class AttemptDispatcher(Protocol):
    @property
    def transport(self) -> DispatchTransport: ...

    async def submit(self, envelope: AttemptEnvelope) -> DispatchReceipt: ...

    async def cancel(self, receipt: DispatchReceipt) -> bool: ...

    async def health(self) -> TransportHealth: ...

    async def close(self) -> None: ...


class PodDispatcher:
    """Submit to the Pod gateway; 202 means the attempt is durably queued in RabbitMQ."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 20.0,
    ) -> None:
        self._base_url = base_url.strip().rstrip("/")
        if not self._base_url or not api_key.strip():
            raise ValueError("pod_dispatcher_requires_base_url_and_api_key")
        self._headers = {"Authorization": f"Bearer {api_key.strip()}"}
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds))
        self._owns_client = client is None

    @property
    def transport(self) -> DispatchTransport:
        return DispatchTransport.POD

    async def submit(self, envelope: AttemptEnvelope) -> DispatchReceipt:
        try:
            response = await self._client.post(
                f"{self._base_url}/v1/attempts",
                json=envelope.model_dump(mode="json"),
                headers=self._headers,
                follow_redirects=False,
            )
        except httpx.TransportError as exc:
            raise DispatchError("pod_unreachable", retryable=True) from exc
        if response.status_code != 202:
            raise DispatchError(
                self._error_code(response),
                status_code=response.status_code,
                retryable=response.status_code in {429, 502, 503, 504},
            )
        body = response.json()
        if body.get("attempt_id") != envelope.attempt_id:
            raise DispatchError("pod_receipt_mismatch", status_code=202, retryable=False)
        return self._receipt(envelope)

    async def cancel(self, receipt: DispatchReceipt) -> bool:
        # Pod attempts are cancelled by the backend's claim decision, not by the transport.
        return False

    async def health(self) -> TransportHealth:
        try:
            response = await self._client.get(f"{self._base_url}/readyz", follow_redirects=False)
            payload = response.json() if response.content else {}
        except (httpx.TransportError, ValueError) as exc:
            return TransportHealth(
                transport=self.transport, ready=False, detail={"error": type(exc).__name__}
            )
        return TransportHealth(
            transport=self.transport,
            ready=response.status_code == 200,
            detail={"status_code": response.status_code, "lanes": payload.get("lanes", {})},
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    @staticmethod
    def _error_code(response: httpx.Response) -> str:
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = None
        return (
            str(detail)
            if isinstance(detail, str) and detail
            else f"pod_http_{response.status_code}"
        )

    def _receipt(self, envelope: AttemptEnvelope) -> DispatchReceipt:
        return DispatchReceipt(
            transport=self.transport,
            job_id=envelope.job_id,
            run_id=envelope.run_id,
            attempt_id=envelope.attempt_id,
            track_id=envelope.track_id,
            source_revision=envelope.source.revision,
        )


class ServerlessDispatcher:
    """Submit to RunPod Serverless endpoints, one endpoint per worker role."""

    DEFAULT_BASE_URL = "https://api.runpod.ai/v2"

    def __init__(
        self,
        endpoints: dict[WorkerRole, str],
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not endpoints or any(not value.strip() for value in endpoints.values()):
            raise ValueError("serverless_dispatcher_requires_endpoint_ids")
        if not api_key.strip():
            raise ValueError("serverless_dispatcher_requires_api_key")
        self._endpoints = {WorkerRole(role): value.strip() for role, value in endpoints.items()}
        self._base_url = base_url.strip().rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key.strip()}"}
        self._topology = RabbitMQTopology()
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds))
        self._owns_client = client is None

    @property
    def transport(self) -> DispatchTransport:
        return DispatchTransport.SERVERLESS

    def endpoint_for(self, envelope: AttemptEnvelope) -> str:
        role = self._topology.role_for(envelope, set(self._endpoints))
        if role is None or role not in self._endpoints:
            raise DispatchError("serverless_endpoint_not_configured_for_job", retryable=False)
        return self._endpoints[role]

    async def submit(self, envelope: AttemptEnvelope) -> DispatchReceipt:
        endpoint = self.endpoint_for(envelope)
        payload = await self._post(f"{endpoint}/run", {"input": envelope.model_dump(mode="json")})
        provider_job_id = str(payload.get("id") or "")
        if not provider_job_id:
            raise DispatchError("serverless_receipt_missing_job_id", retryable=False)
        return DispatchReceipt(
            transport=self.transport,
            job_id=envelope.job_id,
            run_id=envelope.run_id,
            attempt_id=envelope.attempt_id,
            track_id=envelope.track_id,
            source_revision=envelope.source.revision,
            provider_job_id=provider_job_id,
        )

    async def status(self, receipt: DispatchReceipt, endpoint_id: str) -> ServerlessJobStatus:
        if not receipt.provider_job_id:
            raise DispatchError("serverless_receipt_missing_job_id", retryable=False)
        job_id = quote(receipt.provider_job_id, safe="")
        payload = await self._get(f"{endpoint_id}/status/{job_id}")
        state = str(payload.get("status") or "UNKNOWN")
        if state not in {
            "IN_QUEUE",
            "IN_PROGRESS",
            "COMPLETED",
            "FAILED",
            "CANCELLED",
            "TIMED_OUT",
        }:
            state = "UNKNOWN"
        output = payload.get("output")
        if isinstance(output, dict):
            events: tuple[dict[str, Any], ...] = (output,)
        elif isinstance(output, list):
            events = tuple(item for item in output if isinstance(item, dict))
        else:
            events = ()
        error = payload.get("error")
        return ServerlessJobStatus(
            provider_job_id=receipt.provider_job_id,
            state=state,  # type: ignore[arg-type]
            events=events,
            error=str(error) if error else None,
        )

    async def cancel(self, receipt: DispatchReceipt) -> bool:
        if not receipt.provider_job_id:
            return False
        job_id = quote(receipt.provider_job_id, safe="")
        for endpoint in set(self._endpoints.values()):
            try:
                payload = await self._post(f"{endpoint}/cancel/{job_id}", {})
            except DispatchError as exc:
                if exc.status_code == 404:
                    continue
                raise
            if str(payload.get("status") or "").upper() == "CANCELLED":
                return True
        return False

    async def health(self) -> TransportHealth:
        detail: dict[str, Any] = {}
        ready = True
        for role, endpoint in sorted(self._endpoints.items(), key=lambda item: item[0].value):
            try:
                detail[role.value] = await self._get(f"{endpoint}/health")
            except DispatchError as exc:
                ready = False
                detail[role.value] = {"error": exc.code, "status_code": exc.status_code}
        return TransportHealth(transport=self.transport, ready=ready, detail=detail)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._client.post(
                f"{self._base_url}/{path}", json=payload, headers=self._headers
            )
        except httpx.TransportError as exc:
            raise DispatchError("serverless_unreachable", retryable=True) from exc
        return self._payload(response)

    async def _get(self, path: str) -> dict[str, Any]:
        try:
            response = await self._client.get(f"{self._base_url}/{path}", headers=self._headers)
        except httpx.TransportError as exc:
            raise DispatchError("serverless_unreachable", retryable=True) from exc
        return self._payload(response)

    @staticmethod
    def _payload(response: httpx.Response) -> dict[str, Any]:
        if response.status_code >= 400:
            raise DispatchError(
                f"serverless_http_{response.status_code}",
                status_code=response.status_code,
                retryable=response.status_code in {408, 429, 500, 502, 503, 504},
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise DispatchError("serverless_invalid_response", retryable=False) from exc
        if not isinstance(payload, dict):
            raise DispatchError("serverless_invalid_response", retryable=False)
        return payload
