from __future__ import annotations

import asyncio
import os

import uvicorn

from hear.api.app import RuntimeApi
from hear.bootstrap import RuntimeBootstrap
from hear.config import RuntimeSettings
from hear.runtime.pod import PodRuntime


class PodEntrypoint:
    def __init__(
        self,
        bootstrap: RuntimeBootstrap | None = None,
        environment: dict[str, str] | None = None,
    ) -> None:
        source = dict(os.environ) if environment is None else environment
        self._settings = RuntimeSettings.from_environment(source)
        self._bootstrap = bootstrap or RuntimeBootstrap(source)

    async def run(self) -> None:
        role = self._settings.worker_role
        api_key = self._settings.required("pod_api_key")
        readiness = self._bootstrap.readiness(role)
        executor, backend, resources = self._bootstrap.executor_for(role)
        runtime = PodRuntime(
            readiness,
            role,
            executor,
            backend,
            api_key=api_key,
            max_concurrent_jobs=self._settings.pod_max_concurrent_jobs,
            rabbitmq_url=self._settings.required("rabbitmq_url"),
        )
        try:
            server = uvicorn.Server(
                uvicorn.Config(
                    RuntimeApi(
                        readiness,
                        drain=runtime.drain,
                        pod_runtime=runtime,
                        pod_api_key=api_key,
                        enable_docs=self._settings.enable_docs,
                    ).app,
                    host=self._settings.http_host,
                    port=self._settings.http_port,
                    log_level=self._settings.log_level.lower(),
                )
            )
            await runtime.start()
            await server.serve()
        finally:
            error = None
            try:
                await runtime.close()
            except BaseException as exc:
                error = exc
            for resource in reversed(resources):
                close = getattr(resource, "close", None)
                if close is None:
                    continue
                try:
                    value = close()
                    if asyncio.iscoroutine(value):
                        await value
                except BaseException as exc:
                    if error is None:
                        error = exc
            if error is not None:
                raise error


if __name__ == "__main__":
    asyncio.run(PodEntrypoint().run())
