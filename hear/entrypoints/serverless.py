import asyncio
import inspect
import os

from hear.bootstrap import RuntimeBootstrap
from hear.config import RuntimeSettings
from hear.runtime.serverless import ServerlessRuntime


class ServerlessEntrypoint:
    def __init__(self, bootstrap: RuntimeBootstrap | None = None) -> None:
        self._bootstrap = bootstrap or RuntimeBootstrap()
        self._settings = RuntimeSettings.from_environment(dict(os.environ))

    def run(self) -> None:
        role = self._settings.worker_role
        executor, backend, resources = self._bootstrap.executor_for(role)
        readiness = self._bootstrap.readiness(role)
        try:
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
    async def _close_resources(resources: list[object]) -> None:
        for resource in reversed(resources):
            close = getattr(resource, "close", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result


if __name__ == "__main__":
    ServerlessEntrypoint().run()
