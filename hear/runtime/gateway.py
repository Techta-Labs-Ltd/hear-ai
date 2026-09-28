from __future__ import annotations

import importlib
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, TypedDict

from hear.contracts.jobs import AttemptEnvelope
from hear.queue.topology import QueueBinding, RabbitMQTopology
from hear.runtime.roles import WorkerRole


class LaneStatus(TypedDict):
    status: str
    consumers: int
    queued: int
    queue_capacity: int
    queue: str


class GatewayUnavailable(RuntimeError):
    pass


class GatewayQueueFull(GatewayUnavailable):
    pass


class GatewayDeadlineExpired(GatewayUnavailable):
    pass


@dataclass
class GatewayAttempt:
    envelope: AttemptEnvelope
    queue: Any
    closed: bool = False

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            await self.queue.delete(if_unused=False, if_empty=False)
        except Exception:
            pass


class RabbitMQGateway:
    def __init__(
        self,
        rabbitmq_url: str,
        roles: set[WorkerRole],
        provider=None,
    ) -> None:
        self._rabbitmq_url = rabbitmq_url
        self._roles = roles
        self._provider = provider
        self._topology = RabbitMQTopology()
        self._connection: Any = None
        self._channel: Any = None
        self._status_channel: Any = None
        self._exchange: Any = None
        self._retry_exchange: Any = None
        self._dead_exchange: Any = None
        self._queues: dict[WorkerRole, Any] = {}
        self._draining = False

    def _rabbitmq(self):
        if self._provider is None:
            self._provider = importlib.import_module("aio_pika")
        return self._provider

    async def start(self) -> None:
        self._draining = False
        provider = self._rabbitmq()
        self._connection = await provider.connect_robust(self._rabbitmq_url)
        self._channel = await self._connection.channel(
            publisher_confirms=True, on_return_raises=True
        )
        self._status_channel = await self._connection.channel()
        self._exchange = await self._channel.declare_exchange(
            self._topology.exchange,
            provider.ExchangeType.DIRECT,
            durable=True,
        )
        self._retry_exchange = await self._channel.declare_exchange(
            self._topology.retry_exchange,
            provider.ExchangeType.DIRECT,
            durable=True,
        )
        self._dead_exchange = await self._channel.declare_exchange(
            self._topology.dead_exchange,
            provider.ExchangeType.DIRECT,
            durable=True,
        )
        for role in self._roles:
            binding = self._topology.binding(role)
            queue = await self._channel.declare_queue(
                binding.queue,
                durable=True,
                arguments=self._topology.queue_arguments(binding),
            )
            await queue.bind(self._exchange, routing_key=binding.routing_key)
            self._queues[role] = queue
            retry_queue = await self._channel.declare_queue(
                binding.retry_queue,
                durable=True,
                arguments=self._topology.retry_queue_arguments(binding),
            )
            await retry_queue.bind(self._retry_exchange, routing_key=binding.routing_key)
            dead_queue = await self._channel.declare_queue(
                binding.dead_queue,
                durable=True,
                arguments=self._topology.dead_queue_arguments(),
            )
            await dead_queue.bind(
                self._dead_exchange,
                routing_key=binding.dead_routing_key,
            )

    async def close(self) -> None:
        self._draining = True
        if self._status_channel is not None and not self._status_channel.is_closed:
            await self._status_channel.close()
        if self._channel is not None and not self._channel.is_closed:
            await self._channel.close()
        if self._connection is not None and not self._connection.is_closed:
            await self._connection.close()

    async def _admit(self, envelope: AttemptEnvelope) -> QueueBinding:
        if self._draining:
            raise GatewayUnavailable("gateway_draining")
        if self._channel is None or self._connection is None or self._connection.is_closed:
            raise GatewayUnavailable("rabbitmq_unavailable")
        if datetime.now(UTC) >= envelope.deadline:
            raise GatewayDeadlineExpired("attempt_deadline_exceeded")
        if envelope.storage.expires_at <= datetime.now(UTC):
            raise GatewayDeadlineExpired("storage_grant_expired")
        role = self._topology.role_for(envelope, self._roles)
        if role is None or role not in self._roles:
            raise GatewayUnavailable("job_role_unavailable")
        binding = self._topology.binding(role)
        try:
            queue = await self._status_channel.declare_queue(
                binding.queue, passive=True, robust=False
            )
        except Exception as exc:
            raise GatewayUnavailable("queue_status_unavailable") from exc
        if queue.declaration_result.consumer_count < 1:
            raise GatewayUnavailable("job_worker_not_ready")
        if queue.declaration_result.message_count >= self._topology.max_queue_messages:
            raise GatewayQueueFull("job_queue_full")
        return binding

    async def _publish(
        self, envelope: AttemptEnvelope, binding: QueueBinding, reply_to: str | None
    ) -> None:
        provider = self._rabbitmq()
        try:
            await self._exchange.publish(
                provider.Message(
                    body=envelope.model_dump_json().encode(),
                    content_type="application/json",
                    delivery_mode=provider.DeliveryMode.PERSISTENT,
                    message_id=envelope.attempt_id,
                    correlation_id=envelope.job_id,
                    reply_to=reply_to,
                ),
                routing_key=binding.routing_key,
                mandatory=True,
                timeout=15,
            )
        except Exception as exc:
            # A lost confirmation is ambiguous: the backend retries the SAME
            # attempt identity, whose atomic claim prevents duplicate work.
            raise GatewayUnavailable("publish_not_confirmed_retry_same_attempt") from exc

    async def submit(self, envelope: AttemptEnvelope) -> None:
        binding = await self._admit(envelope)
        await self._publish(envelope, binding, None)

    async def enqueue(self, envelope: AttemptEnvelope) -> GatewayAttempt:
        binding = await self._admit(envelope)
        reply_queue = await self._channel.declare_queue(exclusive=True, auto_delete=True)
        attempt = GatewayAttempt(envelope, reply_queue)
        try:
            await self._publish(envelope, binding, reply_queue.name)
        except BaseException:
            await attempt.close()
            raise
        return attempt

    async def stream(self, attempt: GatewayAttempt) -> AsyncGenerator[dict, None]:
        try:
            async with attempt.queue.iterator() as iterator:
                async for message in iterator:
                    async with message.process():
                        payload = json.loads(message.body)
                    if payload.get("kind") == "end":
                        return
                    yield payload
        finally:
            await attempt.close()

    async def lane_status(self) -> dict[str, LaneStatus]:
        if self._channel is None or self._connection is None or self._connection.is_closed:
            return {
                role.value: {
                    "status": "unavailable",
                    "consumers": 0,
                    "queued": 0,
                    "queue_capacity": self._topology.max_queue_messages,
                    "queue": self._topology.binding(role).queue,
                }
                for role in self._roles
            }
        result: dict[str, LaneStatus] = {}
        for role in sorted(self._roles, key=lambda item: item.value):
            binding = self._topology.binding(role)
            try:
                queue = await self._status_channel.declare_queue(
                    binding.queue,
                    passive=True,
                    robust=False,
                )
                consumers = queue.declaration_result.consumer_count
                queued = queue.declaration_result.message_count
                result[role.value] = {
                    "status": "draining"
                    if self._draining
                    else ("ready" if consumers > 0 else "loading"),
                    "consumers": consumers,
                    "queued": queued,
                    "queue_capacity": self._topology.max_queue_messages,
                    "queue": binding.queue,
                }
            except Exception:
                result[role.value] = {
                    "status": "unavailable",
                    "consumers": 0,
                    "queued": 0,
                    "queue_capacity": self._topology.max_queue_messages,
                    "queue": binding.queue,
                }
        return result

    async def drain(self) -> None:
        self._draining = True
