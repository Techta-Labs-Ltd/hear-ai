from __future__ import annotations

import asyncio
import hmac
import json
from collections.abc import AsyncGenerator, AsyncIterator

from fastapi import APIRouter, Header, HTTPException, status
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from hear.contracts.jobs import AttemptEnvelope
from hear.execution.lease import AttemptLeaseLost
from hear.runtime.attempt_stream import AttemptRejection
from hear.runtime.pod import (
    PodAttempt,
    PodQueuedAttempt,
    PodRuntime,
    PodRuntimeBusy,
    PodRuntimeUnavailable,
)


class JobsRouter:
    def __init__(self, runtime: PodRuntime, api_key: str) -> None:
        self._runtime = runtime
        self._api_key = api_key.strip()
        self.router = APIRouter(tags=["jobs"])
        self.router.add_api_route(
            "/v1/attempts/stream",
            self.stream_attempt,
            methods=["POST"],
        )

    async def stream_attempt(
        self,
        envelope: AttemptEnvelope,
        authorization: str | None = Header(default=None),
    ) -> StreamingResponse:
        self._authenticate(authorization)
        try:
            if self._runtime.uses_rabbitmq:
                queued_attempt = await self._runtime.enqueue_attempt(envelope)
                return StreamingResponse(
                    self._queued_events(queued_attempt),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache, no-transform",
                        "X-Accel-Buffering": "no",
                    },
                    background=BackgroundTask(queued_attempt.close),
                )
            direct_attempt = await self._runtime.prepare_attempt(envelope)
        except PodRuntimeBusy as exc:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=str(exc),
                headers={"Retry-After": "5"},
            ) from exc
        except PodRuntimeUnavailable as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="backend_claim_failed",
            ) from exc

        return StreamingResponse(
            self._events(direct_attempt),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
            background=BackgroundTask(self._runtime.close_attempt, direct_attempt),
        )

    def _authenticate(self, authorization: str | None) -> None:
        if not self._api_key:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="pod_api_key_not_configured",
            )
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not token or not hmac.compare_digest(token, self._api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid_pod_api_key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    async def _events(self, attempt: PodAttempt) -> AsyncIterator[str]:
        if isinstance(attempt.result, AttemptRejection):
            yield self._encode(attempt.result.event, attempt.result.data)
            return
        iterator: AsyncGenerator = self._runtime.stream(attempt)
        next_event: asyncio.Future = asyncio.ensure_future(iterator.__anext__())
        try:
            while True:
                done, _ = await asyncio.wait({next_event}, timeout=15.0)
                if not done:
                    yield ": keep-alive\n\n"
                    continue
                try:
                    event = next_event.result()
                except StopAsyncIteration:
                    return
                yield self._encode(
                    event.event.value,
                    event.model_dump(mode="json"),
                    event.event_id,
                )
                next_event = asyncio.ensure_future(iterator.__anext__())
        except AttemptLeaseLost:
            yield self._encode(
                "error",
                {
                    "job_id": attempt.result.envelope.job_id,
                    "attempt_id": attempt.result.envelope.attempt_id,
                    "error_code": "attempt_lease_lost",
                },
            )
        except Exception:
            yield self._encode(
                "error",
                {
                    "job_id": attempt.result.envelope.job_id,
                    "attempt_id": attempt.result.envelope.attempt_id,
                    "error_code": "worker_execution_failed",
                },
            )
        finally:
            if not next_event.done():
                next_event.cancel()
                await asyncio.gather(next_event, return_exceptions=True)
            await iterator.aclose()
            await self._runtime.close_attempt(attempt)

    async def _queued_events(self, attempt: PodQueuedAttempt) -> AsyncIterator[str]:
        yield self._encode(
            "queued",
            {
                "job_id": attempt.envelope.job_id,
                "attempt_id": attempt.envelope.attempt_id,
                "queue": "accepted",
            },
        )
        iterator: AsyncGenerator = self._runtime.stream_queued(attempt)
        next_event: asyncio.Future = asyncio.ensure_future(iterator.__anext__())
        try:
            while True:
                done, _ = await asyncio.wait({next_event}, timeout=15.0)
                if not done:
                    yield ": keep-alive\n\n"
                    continue
                try:
                    event = next_event.result()
                except StopAsyncIteration:
                    return
                if isinstance(event, AttemptRejection):
                    yield self._encode(event.event, event.data)
                else:
                    yield self._encode(
                        event.event.value,
                        event.model_dump(mode="json"),
                        event.event_id,
                    )
                next_event = asyncio.ensure_future(iterator.__anext__())
        except Exception:
            yield self._encode(
                "error",
                {
                    "job_id": attempt.envelope.job_id,
                    "attempt_id": attempt.envelope.attempt_id,
                    "error_code": "worker_execution_failed",
                },
            )
        finally:
            if not next_event.done():
                next_event.cancel()
                await asyncio.gather(next_event, return_exceptions=True)
            await iterator.aclose()
            await attempt.close()

    @staticmethod
    def _encode(event_name: str, data: dict, event_id: str | None = None) -> str:
        payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        lines = []
        if event_id:
            lines.append(f"id: {event_id}")
        lines.append(f"event: {event_name}")
        lines.append(f"data: {payload}")
        return "\n".join(lines) + "\n\n"
