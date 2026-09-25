from __future__ import annotations

import asyncio
import os

import uvicorn

from hear.api.app import RuntimeApi
from hear.bootstrap import RuntimeBootstrap
from hear.queue.rabbitmq import RabbitMQConsumer
from hear.runtime.pod import PodRuntime
from hear.runtime.roles import WorkerRole


class PodEntrypoint:
    def __init__(
        self,
        bootstrap: RuntimeBootstrap | None = None,
        environment: dict[str, str] | None = None,
    ) -> None:
        self._environment = environment or dict(os.environ)
        self._bootstrap = bootstrap or RuntimeBootstrap(self._environment)

    async def run(self) -> None:
        role = WorkerRole(self._environment.get("HEAR_WORKER_ROLE", "transcription"))
        if role != WorkerRole.TRANSCRIPTION:
            raise RuntimeError("unsupported_entrypoint_role")
        readiness = self._bootstrap.readiness(role)
        executor, backend, resources = self._bootstrap.transcription_executor()
        consumer = RabbitMQConsumer(
            self._required("HEAR_RABBITMQ_URL"),
            role,
            executor,
            backend,
            self._bootstrap.worker_identity(),
        )
        runtime = PodRuntime(readiness, consumer)
        server = uvicorn.Server(
            uvicorn.Config(
                RuntimeApi(readiness).app,
                host=self._environment.get("HTTP_HOST", "0.0.0.0"),
                port=int(self._environment.get("HTTP_PORT", "8000")),
                log_level=self._environment.get("LOG_LEVEL", "info").lower(),
            )
        )
        await runtime.start()
        try:
            await server.serve()
        finally:
            await runtime.close()
            for resource in reversed(resources):
                close = getattr(resource, "close", None)
                if close is None:
                    continue
                value = close()
                if asyncio.iscoroutine(value):
                    await value

    def _required(self, name: str) -> str:
        value = self._environment.get(name, "").strip()
        if not value:
            raise RuntimeError(f"missing_runtime_setting:{name}")
        return value


if __name__ == "__main__":
    asyncio.run(PodEntrypoint().run())