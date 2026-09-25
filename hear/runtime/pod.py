from __future__ import annotations

import asyncio

import aio_pika
from aio_pika.abc import AbstractIncomingMessage, AbstractRobustChannel, AbstractRobustConnection

from hear.contracts.jobs import AttemptEnvelope
from hear.execution.executor import JobExecutor
from hear.execution.reporter import BackendAttemptClient
from hear.runtime.config import RuntimeSettings


class PodRuntime:
    def __init__(
        self,
        settings: RuntimeSettings,
        executor: JobExecutor,
        backend: BackendAttemptClient,
    ) -> None:
        self._settings = settings
        self._executor = executor
        self._backend = backend
        self._connection: AbstractRobustConnection | None = None
        self._channel: AbstractRobustChannel | None = None
        self._consumer_tag: str | None = None
        self._active = 0
        self._draining = False
        self._slot = asyncio.Semaphore(1)

    @property
    def ready(self) -> bool:
        return (
            not self._draining
            and self._connection is not None
            and not self._connection.is_closed
            and self._channel is not None
            and not self._channel.is_closed
        )

    @property
    def active(self) -> int:
        return self._active

    async def start(self) -> None:
        if not self._settings.rabbitmq_url or not self._settings.rabbitmq_queue:
            raise RuntimeError("rabbitmq_configuration_missing")
        self._connection = await aio_pika.connect_robust(self._settings.rabbitmq_url)
        self._channel = await self._connection.channel()
        await self._channel.set_qos(prefetch_count=self._settings.rabbitmq_prefetch)
        queue = await self._channel.declare_queue(
            self._settings.rabbitmq_queue,
            durable=True,
            arguments={"x-queue-type": "quorum"},
        )
        self._consumer_tag = await queue.consume(self._consume)

    async def _consume(self, message: AbstractIncomingMessage) -> None:
        async with self._slot:
            request = AttemptEnvelope.model_validate_json(message.body)
            if request.job_type != self._settings.worker_role:
                await message.reject(requeue=False)
                return
            decision = await self._backend.claim(request)
            if decision != "execute":
                await message.ack()
                return
            await message.ack()
            self._active += 1
            try:
                async for event in self._executor.stream(request):
                    await self._backend.event(request, event)
            finally:
                self._active -= 1

    async def drain(self) -> None:
        self._draining = True
        if self._channel is not None and self._consumer_tag is not None:
            await self._channel.cancel(self._consumer_tag)
            self._consumer_tag = None
        while self._active:
            await asyncio.sleep(0.1)

    async def close(self) -> None:
        await self.drain()
        await self._backend.close()
        if self._channel is not None and not self._channel.is_closed:
            await self._channel.close()
        if self._connection is not None and not self._connection.is_closed:
            await self._connection.close()
