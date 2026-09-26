from __future__ import annotations

import importlib
import json

from hear.contracts.events import ExecutionEventType
from hear.contracts.jobs import AttemptEnvelope, ClaimDecision, WorkerIdentity
from hear.contracts.outcomes import ExecutionOutcome
from hear.execution.executor import JobExecutor
from hear.execution.reporter import BackendAttemptClient
from hear.queue.topology import RabbitMQTopology
from hear.runtime.roles import WorkerCapabilityRegistry, WorkerRole


class RabbitMQConsumer:
    def __init__(
        self,
        connection_url: str,
        role: WorkerRole,
        executor: JobExecutor,
        backend: BackendAttemptClient,
        worker: WorkerIdentity,
        *,
        topology: RabbitMQTopology | None = None,
        prefetch: int = 1,
        provider=None,
    ) -> None:
        self._connection_url = connection_url
        self._role = role
        self._capability = WorkerCapabilityRegistry().get(role)
        self._executor = executor
        self._backend = backend
        self._worker = worker
        self._topology = topology or RabbitMQTopology()
        self._prefetch = max(1, prefetch)
        self._provider = provider
        self._connection = None
        self._channel = None
        self._consumer_tag = None
        self._queue = None

    def _aio_pika(self):
        if self._provider is None:
            self._provider = importlib.import_module("aio_pika")
        return self._provider

    async def start(self) -> None:
        provider = self._aio_pika()
        binding = self._topology.binding(self._role)
        connection = await provider.connect_robust(self._connection_url)
        channel = await connection.channel()
        self._connection = connection
        self._channel = channel
        await channel.set_qos(prefetch_count=self._prefetch)
        exchange = await channel.declare_exchange(
            self._topology.exchange,
            provider.ExchangeType.DIRECT,
            durable=True,
        )
        queue = await channel.declare_queue(
            binding.queue,
            durable=True,
            arguments={"x-queue-type": "quorum"},
        )
        self._queue = queue
        await queue.bind(exchange, routing_key=binding.routing_key)
        self._consumer_tag = await queue.consume(self._on_message, no_ack=False)

    async def close(self) -> None:
        if self._queue is not None and self._consumer_tag is not None:
            await self._queue.cancel(self._consumer_tag)
        if self._channel is not None:
            await self._channel.close()
        if self._connection is not None:
            await self._connection.close()
        await self._backend.close()

    async def _on_message(self, message) -> None:
        try:
            envelope = AttemptEnvelope.model_validate(json.loads(message.body))
        except Exception:
            await message.reject(requeue=False)
            return
        if not self._capability.accepts(envelope):
            await message.reject(requeue=False)
            return
        try:
            decision = await self._backend.claim(envelope)
        except Exception:
            await message.nack(requeue=True)
            return
        await message.ack()
        if decision != ClaimDecision.EXECUTE:
            return
        try:
            async for event in self._executor.stream(envelope):
                if event.event == ExecutionEventType.OUTCOME:
                    outcome = ExecutionOutcome.model_validate(event.data.get("outcome"))
                    await self._backend.outcome(envelope, outcome)
                else:
                    await self._backend.event(envelope, event)
        except Exception:
            outcome = ExecutionOutcome(
                job_id=envelope.job_id,
                attempt_id=envelope.attempt_id,
                track_id=envelope.track_id,
                job_type=envelope.job_type,
                source_revision=envelope.source.revision,
                status="failed",
                error_code="worker_execution_failed",
            )
            try:
                await self._backend.outcome(envelope, outcome)
            except Exception:
                return
