import asyncio
import inspect
import os
import time

from hear.bootstrap import RuntimeBootstrap
from hear.config import RuntimeSettings
from hear.runtime.serverless import ServerlessRuntime


class ServerlessEntrypoint:
    def __init__(self, bootstrap: RuntimeBootstrap | None = None) -> None:
        self._bootstrap = bootstrap or RuntimeBootstrap()
        self._settings = RuntimeSettings.from_environment(dict(os.environ))

    def run(self) -> None:
        role = self._settings.worker_role
        if self._settings.serverless_preload_models and self._settings.gpu_idle_eviction_enabled:
            raise RuntimeError("serverless_preload_requires_idle_eviction_disabled")
        executor, backend, resources = self._bootstrap.executor_for(role)
        readiness = self._bootstrap.readiness(role)
        try:
            if not readiness.is_ready():
                raise RuntimeError("runtime_not_ready")
            if self._settings.serverless_preload_models:
                asyncio.run(self._warm_resources(resources))
                if not readiness.is_ready():
                    raise RuntimeError("runtime_not_ready")
            ServerlessRuntime(
                role,
                executor,
                backend,
                readiness=readiness,
                max_concurrent_jobs=self._settings.serverless_max_concurrent_jobs,
            ).start()
        finally:
            asyncio.run(self._close_resources(resources))

    @staticmethod
    async def _warm_resources(resources: list[object]) -> None:
        # The SDK creates its own event loops after startup. Preloaded models
        # retain weights without scheduling idle timers on this temporary loop.
        for resource in resources:
            warmup = getattr(resource, "warmup", None)
            if callable(warmup):
                started = time.perf_counter()
                result = warmup()
                if inspect.isawaitable(result):
                    await result
                print(
                    f"serverless_model_preloaded:{type(resource).__name__}:"
                    f"{time.perf_counter() - started:.3f}s",
                    flush=True,
                )

    @staticmethod
    async def _close_resources(resources: list[object]) -> None:
        for resource in reversed(resources):
            close = getattr(resource, "close", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result


if __name__ == "__main__":
    ServerlessEntrypoint().run()
