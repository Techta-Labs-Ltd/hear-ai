from __future__ import annotations

import asyncio
import importlib
import json
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

from hear.contracts.events import ExecutionEvent, ExecutionEventType
from hear.contracts.jobs import AttemptEnvelope
from hear.contracts.outcomes import ExecutionOutcome
from hear.execution.executor import FailureSummary, JobExecutor
from hear.execution.reporter import BackendAttemptClient
from hear.health.service import RuntimeReadiness
from hear.queue.topology import RabbitMQTopology
from hear.runtime.attempt_stream import (
    AttemptDeadlineExceeded,
    AttemptRejection,
    AttemptStream,
    PreparedAttempt,
)
from hear.runtime.host_admission import HostJobAdmission, HostJobPermit
from hear.runtime.roles import WorkerRole


class PodRuntimeUnavailable(RuntimeError):
    pass


class PodRuntimeBusy(RuntimeError):
    pass


@dataclass
class PodAttempt:
    result: PreparedAttempt | AttemptRejection
    counted: bool
    runtime: PodRuntime
    permit: HostJobPermit | None = None
    closed: bool = False

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if isinstance(self.result, PreparedAttempt):
                await self.result.close()
        finally:
            if self.counted:
                await self.runtime._release_attempt()
            if self.permit is not None:
                self.permit.close()


@dataclass
class PodQueuedAttempt:
    envelope: AttemptEnvelope
    events: asyncio.Queue[ExecutionEvent | AttemptRejection | None]
    runtime: PodRuntime
    closed: bool = False

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        await self.runtime._remove_subscriber(self.envelope.attempt_id, self.events)


class PodRuntime:
    def __init__(
        self,
        readiness: RuntimeReadiness,
        role: WorkerRole,
        executor: JobExecutor,
        backend: BackendAttemptClient,
        *,
        api_key: str,
        max_concurrent_jobs: int = 1,
        rabbitmq_url: str | None = None,
        rabbitmq_provider=None,
        require_api_key: bool = True,
        host_admission: HostJobAdmission | None = None,
    ) -> None:
        self._readiness = readiness
        self._role = role
        self._backend = backend
        self._attempt_stream = AttemptStream(role, executor, backend)
        self._api_key = api_key.strip()
        self._max_concurrent_jobs = max_concurrent_jobs
        self._rabbitmq_url = (rabbitmq_url or "").strip()
        self._rabbitmq_provider: Any = rabbitmq_provider
        self._rabbitmq_connection: Any = None
        self._rabbitmq_channel: Any = None
        self._rabbitmq_queue: Any = None
        self._rabbitmq_consumer_tag: Any = None
        self._rabbitmq_exchange: Any = None
        self._rabbitmq_retry_exchange: Any = None
        self._rabbitmq_dead_exchange: Any = None
        self._topology = RabbitMQTopology()
        self._host_admission = host_admission
        self._subscribers: dict[
            str,
            set[asyncio.Queue[ExecutionEvent | AttemptRejection | None]],
        ] = {}
        self._active_jobs = 0
        self._started = False
        self._draining = False
        self._lock = asyncio.Lock()
        self._idle = asyncio.Event()
        self._idle.set()
        self._fatal = asyncio.Event()
        if require_api_key:
            readiness.add_check("pod_api_key", lambda: bool(self._api_key))
        readiness.add_check(
            "pod_http_admission" if require_api_key else "queue_admission",
            self._accepting,
        )
        if self._rabbitmq_url:
            readiness.add_check("rabbitmq", self._rabbitmq_ready)

    def _accepting(self) -> bool:
        return self._started and not self._draining

    @property
    def uses_rabbitmq(self) -> bool:
        return bool(self._rabbitmq_url)

    def _rabbitmq_ready(self) -> bool:
        return bool(
            self._rabbitmq_connection is not None
            and not self._rabbitmq_connection.is_closed
            and self._rabbitmq_queue is not None
        )

    def _rabbitmq(self) -> Any:
        if self._rabbitmq_provider is None:
            self._rabbitmq_provider = importlib.import_module("aio_pika")
        return self._rabbitmq_provider

    async def start(self) -> None:
        self._draining = False
        if self._rabbitmq_url:
            provider = self._rabbitmq()
            binding = self._topology.binding(self._role)
            self._rabbitmq_connection = await provider.connect_robust(self._rabbitmq_url)
            self._rabbitmq_channel = await self._rabbitmq_connection.channel()
            await self._rabbitmq_channel.set_qos(prefetch_count=self._max_concurrent_jobs)
            self._rabbitmq_exchange = await self._rabbitmq_channel.declare_exchange(
                self._topology.exchange,
                provider.ExchangeType.DIRECT,
                durable=True,
            )
            self._rabbitmq_retry_exchange = await self._rabbitmq_channel.declare_exchange(
                self._topology.retry_exchange,
                provider.ExchangeType.DIRECT,
                durable=True,
            )
            self._rabbitmq_dead_exchange = await self._rabbitmq_channel.declare_exchange(
                self._topology.dead_exchange,
                provider.ExchangeType.DIRECT,
                durable=True,
            )
            self._rabbitmq_queue = await self._rabbitmq_channel.declare_queue(
                binding.queue,
                durable=True,
                arguments=self._topology.queue_arguments(binding),
            )
            await self._rabbitmq_queue.bind(
                self._rabbitmq_exchange,
                routing_key=binding.routing_key,
            )
            retry_queue = await self._rabbitmq_channel.declare_queue(
                binding.retry_queue,
                durable=True,
                arguments=self._topology.retry_queue_arguments(binding),
            )
            await retry_queue.bind(
                self._rabbitmq_retry_exchange,
                routing_key=binding.routing_key,
            )
            dead_queue = await self._rabbitmq_channel.declare_queue(
                binding.dead_queue,
                durable=True,
                arguments=self._topology.dead_queue_arguments(),
            )
            await dead_queue.bind(
                self._rabbitmq_dead_exchange,
                routing_key=binding.dead_routing_key,
            )
            self._rabbitmq_consumer_tag = await self._rabbitmq_queue.consume(
                self._on_rabbitmq_message,
                no_ack=False,
            )
        self._started = True
        self._readiness.set_draining(False)
        if not self._readiness.is_ready():
            self._started = False
            raise RuntimeError("runtime_not_ready")

    async def enqueue_attempt(self, envelope: AttemptEnvelope) -> PodQueuedAttempt:
        if not self.uses_rabbitmq:
            raise PodRuntimeUnavailable("rabbitmq_not_configured")
        async with self._lock:
            if (
                not self._accepting()
                or not self._readiness.is_ready()
                or not self._rabbitmq_ready()
            ):
                raise PodRuntimeUnavailable("runtime_not_ready")
            subscriber: asyncio.Queue[ExecutionEvent | AttemptRejection | None] = asyncio.Queue()
            self._subscribers.setdefault(envelope.attempt_id, set()).add(subscriber)
        provider = self._rabbitmq()
        try:
            message = provider.Message(
                body=envelope.model_dump_json().encode(),
                content_type="application/json",
                delivery_mode=provider.DeliveryMode.PERSISTENT,
                message_id=envelope.attempt_id,
                correlation_id=envelope.job_id,
            )
            binding = self._topology.binding(self._role)
            await self._rabbitmq_exchange.publish(message, routing_key=binding.routing_key)
        except Exception:
            await self._remove_subscriber(envelope.attempt_id, subscriber)
            raise
        return PodQueuedAttempt(envelope, subscriber, self)

    async def stream_queued(
        self,
        attempt: PodQueuedAttempt,
    ) -> AsyncGenerator[ExecutionEvent | AttemptRejection, None]:
        try:
            while True:
                event = await attempt.events.get()
                if event is None:
                    return
                yield event
        finally:
            await attempt.close()

    async def prepare_attempt(self, envelope: AttemptEnvelope) -> PodAttempt:
        permit = self._host_admission.try_acquire() if self._host_admission is not None else None
        if self._host_admission is not None and permit is None:
            raise PodRuntimeBusy("host_capacity_reached")
        async with self._lock:
            if not self._accepting() or not self._readiness.is_ready():
                if permit is not None:
                    permit.close()
                raise PodRuntimeUnavailable("runtime_not_ready")
            if self._active_jobs >= self._max_concurrent_jobs:
                if permit is not None:
                    permit.close()
                raise PodRuntimeBusy("pod_capacity_reached")
            self._active_jobs += 1
            self._idle.clear()
        try:
            result = await self._attempt_stream.prepare(envelope)
        except Exception:
            await self._release_attempt()
            if permit is not None:
                permit.close()
            raise
        if isinstance(result, AttemptRejection):
            await self._release_attempt()
            if permit is not None:
                permit.close()
            return PodAttempt(result, False, self)
        return PodAttempt(result, True, self, permit)

    async def stream(self, attempt: PodAttempt) -> AsyncGenerator[ExecutionEvent, None]:
        if isinstance(attempt.result, AttemptRejection):
            return
        try:
            async for event in self._attempt_stream.stream(attempt.result):
                yield event
        finally:
            await attempt.close()

    async def drain(self) -> None:
        async with self._lock:
            self._draining = True
            self._readiness.set_draining(True)
        if self._rabbitmq_queue is not None and self._rabbitmq_consumer_tag is not None:
            await self._rabbitmq_queue.cancel(self._rabbitmq_consumer_tag)
            self._rabbitmq_consumer_tag = None
        await self._idle.wait()

    async def close(self) -> None:
        await self.drain()
        if self._rabbitmq_channel is not None and not self._rabbitmq_channel.is_closed:
            await self._rabbitmq_channel.close()
        if self._rabbitmq_connection is not None and not self._rabbitmq_connection.is_closed:
            await self._rabbitmq_connection.close()

    async def close_attempt(self, attempt: PodAttempt) -> None:
        await attempt.close()

    async def wait_until_unhealthy(self) -> None:
        await self._fatal.wait()

    async def _release_attempt(self) -> None:
        async with self._lock:
            self._active_jobs = max(0, self._active_jobs - 1)
            if self._active_jobs == 0:
                self._idle.set()

    async def _remove_subscriber(
        self,
        attempt_id: str,
        subscriber: asyncio.Queue[ExecutionEvent | AttemptRejection | None],
    ) -> None:
        async with self._lock:
            listeners = self._subscribers.get(attempt_id)
            if listeners is None:
                return
            listeners.discard(subscriber)
            if not listeners:
                self._subscribers.pop(attempt_id, None)

    async def _publish_local(
        self,
        attempt_id: str,
        event: ExecutionEvent | AttemptRejection | None,
        reply_to: str | None = None,
    ) -> None:
        listeners = tuple(self._subscribers.get(attempt_id, ()))
        for listener in listeners:
            await listener.put(event)
        if reply_to and self._rabbitmq_channel is not None:
            if isinstance(event, ExecutionEvent):
                payload = {"kind": "event", "data": event.model_dump(mode="json")}
            elif isinstance(event, AttemptRejection):
                payload = {"kind": "rejection", "event": event.event, "data": event.data}
            else:
                payload = {"kind": "end"}
            provider = self._rabbitmq()
            try:
                await self._rabbitmq_channel.default_exchange.publish(
                    provider.Message(
                        body=json.dumps(payload, separators=(",", ":")).encode(),
                        content_type="application/json",
                        correlation_id=attempt_id,
                    ),
                    routing_key=reply_to,
                    mandatory=False,
                    timeout=5,
                )
            except Exception:
                # The browser/request may already have disconnected. The backend
                # callback, not an ephemeral preview queue, is authoritative.
                logging.getLogger(__name__).warning("Optional attempt preview delivery failed")

    async def _on_rabbitmq_message(self, message) -> None:
        try:
            envelope = AttemptEnvelope.model_validate_json(message.body)
        except Exception:
            await message.reject(requeue=False)
            return
        try:
            attempt = await self.prepare_attempt(envelope)
        except PodRuntimeBusy:
            dead, retry_count = await self._retry_or_dead_letter(
                message, capacity_wait=True, expired=envelope.deadline <= datetime.now(UTC)
            )
            await self._publish_local(
                envelope.attempt_id,
                AttemptRejection(
                    "worker_waiting" if not dead else "attempt_dead_lettered",
                    {
                        "job_id": envelope.job_id,
                        "attempt_id": envelope.attempt_id,
                        "retry_count": retry_count,
                    },
                ),
                message.reply_to,
            )
            if dead:
                await self._publish_local(envelope.attempt_id, None, message.reply_to)
            return
        except Exception:
            dead, retry_count = await self._retry_or_dead_letter(message)
            await self._publish_local(
                envelope.attempt_id,
                AttemptRejection(
                    "worker_retrying" if not dead else "attempt_dead_lettered",
                    {
                        "job_id": envelope.job_id,
                        "attempt_id": envelope.attempt_id,
                        "retry_count": retry_count,
                    },
                ),
                message.reply_to,
            )
            if dead:
                await self._publish_local(envelope.attempt_id, None, message.reply_to)
            return
        reply_to = message.reply_to
        if isinstance(attempt.result, AttemptRejection):
            # A redelivery can arrive before a disconnected worker's backend
            # lease expires. Do not acknowledge away that recoverable job.
            if attempt.result.data.get("decision") == "lease_unavailable":
                dead, _ = await self._retry_or_dead_letter(
                    message, capacity_wait=True, expired=envelope.deadline <= datetime.now(UTC)
                )
                await self._publish_local(envelope.attempt_id, attempt.result, reply_to)
                if dead:
                    await self._publish_local(envelope.attempt_id, None, reply_to)
                return
            await self._publish_local(envelope.attempt_id, attempt.result, reply_to)
            await self._publish_local(envelope.attempt_id, None, reply_to)
            await message.ack()
            return
        failure: Exception | None = None
        try:
            async for event in self.stream(attempt):
                if event.event == ExecutionEventType.OUTCOME:
                    outcome = ExecutionOutcome.model_validate(event.data.get("outcome"))
                    await self._report_with_retry(self._backend.outcome, envelope, outcome)
                else:
                    report_event = getattr(self._backend, "event", None)
                    if callable(report_event):
                        await self._report_with_retry(report_event, envelope, event)
                await self._publish_local(envelope.attempt_id, event, reply_to)
        except Exception as exc:
            failure = exc
        if failure is None:
            await self._publish_local(envelope.attempt_id, None, reply_to)
            await message.ack()
            return
        if isinstance(failure, AttemptDeadlineExceeded):
            await self._publish_local(
                envelope.attempt_id,
                AttemptRejection(
                    "attempt_deadline_exceeded",
                    {"job_id": envelope.job_id, "attempt_id": envelope.attempt_id},
                ),
                reply_to,
            )
            await self._publish_local(envelope.attempt_id, None, reply_to)
            await message.ack()
            return
        if (
            isinstance(failure, httpx.HTTPStatusError)
            and failure.response.status_code in BackendAttemptClient.REJECTED_STATUSES
        ):
            await self._publish_local(
                envelope.attempt_id,
                AttemptRejection(
                    "backend_attempt_rejected",
                    {"job_id": envelope.job_id, "attempt_id": envelope.attempt_id},
                ),
                reply_to,
            )
            await self._publish_local(envelope.attempt_id, None, reply_to)
            await message.ack()
            return
        dead, retry_count = await self._retry_or_dead_letter(message)
        await self._publish_local(
            envelope.attempt_id,
            AttemptRejection(
                "worker_retrying" if not dead else "worker_execution_failed",
                {
                    "job_id": envelope.job_id,
                    "attempt_id": envelope.attempt_id,
                    "retry_count": retry_count,
                    "error": FailureSummary.describe(failure),
                },
            ),
            reply_to,
        )
        if dead:
            await self._publish_local(envelope.attempt_id, None, reply_to)
        if not self._readiness.is_ready():
            self._fatal.set()

    async def _retry_or_dead_letter(
        self, message, *, capacity_wait: bool = False, expired: bool = False
    ) -> tuple[bool, int]:
        binding = self._topology.binding(self._role)
        headers = dict(message.headers or {})
        retry_count = int(headers.get("x-hear-retry-count") or 0) + (0 if capacity_wait else 1)
        headers["x-hear-retry-count"] = retry_count
        dead = expired or retry_count > self._topology.max_retries
        exchange = self._rabbitmq_dead_exchange if dead else self._rabbitmq_retry_exchange
        routing_key = binding.dead_routing_key if dead else binding.routing_key
        provider = self._rabbitmq()
        await exchange.publish(
            provider.Message(
                body=message.body,
                content_type=message.content_type or "application/json",
                delivery_mode=provider.DeliveryMode.PERSISTENT,
                message_id=message.message_id,
                correlation_id=message.correlation_id,
                reply_to=message.reply_to,
                headers=headers,
            ),
            routing_key=routing_key,
        )
        await message.ack()
        return dead, retry_count

    @staticmethod
    async def _report_with_retry(callback, *args) -> None:
        last_error: Exception | None = None
        for delay in (0.0, 1.0, 2.0, 4.0):
            if delay:
                await asyncio.sleep(delay)
            try:
                await callback(*args)
                return
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500 and exc.response.status_code != 429:
                    raise
                last_error = exc
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
