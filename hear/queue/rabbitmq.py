from __future__ import annotations

import asyncio
import importlib
import json

from hear.contracts.events import ExecutionEventType
from hear.contracts.jobs import AttemptEnvelope, ClaimDecision, WorkerIdentity
from hear.contracts.outcomes import ExecutionOutcome
from hear.execution.executor import JobExecutor
from hear.execution.lease import AttemptLease, AttemptLeaseLost
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
        self._active = 0
        self._idle = asyncio.Event()
        self._idle.set()

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

    async def drain(self) -> None:
        if self._queue is not None and self._consumer_tag is not None:
            await self._queue.cancel(self._consumer_tag)
            self._consumer_tag = None
        await self._idle.wait()

    async def close(self) -> None:
        await self.drain()
        if self._channel is not None:
            await self._channel.close()
        if self._connection is not None:
            await self._connection.close()
        await self._backend.close()

    async def _on_message(self, message) -> None:
        self._active += 1
        self._idle.clear()
        try:
            await self._handle_message(message)
        finally:
            self._active -= 1
            if self._active == 0:
                self._idle.set()

    async def _handle_message(self, message) -> None:
        try:
            envelope = AttemptEnvelope.model_validate(json.loads(message.body))
        except Exception:
            await message.reject(requeue=False)
            return
        if not self._capability.accepts(envelope):
            await message.reject(requeue=False)
            return
        try:
            claim = await self._backend.claim(envelope)
        except Exception:
            await message.nack(requeue=True)
            return
        await message.ack()
        if claim.decision != ClaimDecision.EXECUTE:
            return
        lease = AttemptLease(self._backend, envelope, claim)
        lease.start()
        iterator = self._executor.stream(envelope).__aiter__()
        try:
            while True:
                try:
                    event = await lease.next_event(iterator)
                except StopAsyncIteration:
                    break
                if event.event == ExecutionEventType.OUTCOME:
                    outcome = ExecutionOutcome.model_validate(event.data.get("outcome"))
                    try:
                        await self._backend.outcome(envelope, outcome)
                    except Exception:
                        return
                else:
                    try:
                        await self._backend.event(envelope, event)
                    except Exception:
                        pass
        except AttemptLeaseLost:
            return
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
                pass
        finally:
            await lease.close()
            close = getattr(iterator, "aclose", None)
            if callable(close):
                await close()
