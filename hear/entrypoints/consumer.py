from __future__ import annotations

import asyncio
import os
import signal

from hear.bootstrap import RuntimeBootstrap
from hear.config import RuntimeSettings
from hear.runtime.host_admission import HostJobAdmission
from hear.runtime.pod import PodRuntime
from hear.runtime.roles import WorkerRole


class ConsumerEntrypoint:
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
        if role == WorkerRole.RECONSTRUCTION and self._settings.pod_max_concurrent_jobs != 1:
            raise ValueError("fish_reconstruction_requires_one_job_per_worker")
        readiness = self._bootstrap.readiness(role)
        executor, backend, resources = self._bootstrap.executor_for(role)
        runtime = PodRuntime(
            readiness,
            role,
            executor,
            backend,
            api_key="",
            max_concurrent_jobs=self._settings.pod_max_concurrent_jobs,
            rabbitmq_url=self._settings.required("rabbitmq_url"),
            require_api_key=False,
            host_admission=HostJobAdmission(
                self._settings.host_job_lock_path,
                self._settings.host_max_concurrent_jobs,
                role=role.value,
                role_limit=1,
            ),
        )
        stopped = asyncio.Event()
        loop = asyncio.get_running_loop()
        for event in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(event, stopped.set)
        try:
            await runtime.start()
            signal_waiter = asyncio.create_task(stopped.wait())
            health_waiter = asyncio.create_task(runtime.wait_until_unhealthy())
            done, pending = await asyncio.wait(
                {signal_waiter, health_waiter},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            if health_waiter in done:
                raise RuntimeError("runtime_became_unhealthy")
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
    asyncio.run(ConsumerEntrypoint().run())
