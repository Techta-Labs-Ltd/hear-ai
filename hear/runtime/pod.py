from __future__ import annotations

from hear.health.service import RuntimeReadiness
from hear.queue.rabbitmq import RabbitMQConsumer


class PodRuntime:
    def __init__(
        self,
        readiness: RuntimeReadiness,
        consumer: RabbitMQConsumer,
    ) -> None:
        self._readiness = readiness
        self._consumer = consumer
        self._started = False

    async def start(self) -> None:
        self._readiness.initialize()
        if not self._readiness.is_ready():
            raise RuntimeError("runtime_not_ready")
        await self._consumer.start()
        self._started = True

    async def close(self) -> None:
        if self._started:
            await self._consumer.close()
            self._started = False