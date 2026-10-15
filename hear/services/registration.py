"""Pod self-registration: the worker tells the backend where it is, not the other way round.

A Pod gets a new ID whenever it is replaced, and a backend that holds the old
gateway URL in config keeps dispatching into the void until someone edits it.
The gateway therefore announces itself: every `interval_seconds` it POSTs its
public URL, roles, lane status and image identity to the backend's worker
registry, and on shutdown it announces that it is going offline. The backend
dispatches to the registry, never to a configured URL, and treats a worker it
has not heard from for a few intervals as gone.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import Any

import httpx

HEARTBEAT_PATH = "/internal/ai/runtime/workers/heartbeat"
SCHEMA_VERSION = 1


class WorkerRegistrar:
    def __init__(
        self,
        *,
        backend_url: str,
        service_key: str,
        gateway_url: str,
        worker_id: str,
        worker_generation: str,
        roles: list[str],
        image_revision: str | None,
        engine_revision: str | None,
        lane_status: Callable[[], Awaitable[Mapping[str, Any]]],
        interval_seconds: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("registration_interval_must_be_positive")
        self._url = f"{backend_url.rstrip('/')}{HEARTBEAT_PATH}"
        self._headers = {"X-Service-Key": service_key}
        self._gateway_url = gateway_url.rstrip("/")
        self._identity = {
            "worker_id": worker_id,
            "worker_generation": worker_generation,
            "roles": sorted(roles),
            "image_revision": image_revision,
            "engine_revision": engine_revision,
        }
        self._lane_status = lane_status
        self._interval = interval_seconds
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(15.0))
        self._owns_client = client is None
        self._task: asyncio.Task | None = None
        self._last_outcome: str | None = None
        self.sent = 0
        self.failures = 0

    @staticmethod
    def derive_gateway_url(environment: Mapping[str, str]) -> str | None:
        """Explicit `HEAR_PUBLIC_GATEWAY_URL`, else the RunPod proxy URL for this Pod."""
        explicit = (environment.get("HEAR_PUBLIC_GATEWAY_URL") or "").strip()
        if explicit:
            return explicit.rstrip("/")
        pod_id = (environment.get("RUNPOD_POD_ID") or "").strip()
        if not pod_id:
            return None
        port = (environment.get("HTTP_PORT") or "8000").strip()
        return f"https://{pod_id}-{port}.proxy.runpod.net"

    async def payload(self, status: str | None = None) -> dict[str, Any]:
        lanes = dict(await self._lane_status())
        if status is None:
            if lanes and all(item.get("status") == "ready" for item in lanes.values()):
                status = "ready"
            elif any(item.get("status") == "draining" for item in lanes.values()):
                status = "draining"
            else:
                status = "loading"
        return {
            "schema_version": SCHEMA_VERSION,
            "transport": "pod",
            **self._identity,
            "gateway_url": self._gateway_url,
            "capabilities_url": f"{self._gateway_url}/capabilities",
            "status": status,
            "lanes": lanes,
            "interval_seconds": self._interval,
            "sent_at": datetime.now(UTC).isoformat(),
        }

    async def send(self, status: str | None = None) -> bool:
        body = await self.payload(status)
        try:
            response = await self._client.post(self._url, json=body, headers=self._headers)
        except httpx.HTTPError as exc:
            self._note("failed", f"{type(exc).__name__}")
            return False
        if response.status_code >= 400:
            self._note("failed", f"http_{response.status_code}")
            return False
        self.sent += 1
        self._note("sent", body["status"])
        return True

    def _note(self, outcome: str, detail: str) -> None:
        logger = logging.getLogger(__name__)
        key = f"{outcome}:{detail}"
        if outcome == "failed":
            self.failures += 1
        if key != self._last_outcome:
            # Log state changes only; a healthy loop would otherwise log every 30 s.
            level = logging.WARNING if outcome == "failed" else logging.INFO
            logger.log(
                level,
                "worker_registration outcome=%s detail=%s url=%s",
                outcome,
                detail,
                self._gateway_url,
            )
        self._last_outcome = key

    async def run(self) -> None:
        delay = self._interval
        while True:
            ok = await self.send()
            # Back off while the backend is unreachable, never past four intervals.
            delay = self._interval if ok else min(delay * 2, self._interval * 4)
            await asyncio.sleep(delay)

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self.run(), name="worker-registration")

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        try:
            await asyncio.wait_for(self.send("offline"), timeout=5.0)
        except Exception:
            pass
        if self._owns_client:
            await self._client.aclose()
