import os

from hear.bootstrap import RuntimeBootstrap
from hear.runtime.roles import WorkerRole
from hear.runtime.serverless import ServerlessRuntime


class ServerlessEntrypoint:
    def __init__(self, bootstrap: RuntimeBootstrap | None = None) -> None:
        self._bootstrap = bootstrap or RuntimeBootstrap()

    def run(self) -> None:
        role = WorkerRole(os.environ.get("HEAR_WORKER_ROLE", "transcription"))
        if role == WorkerRole.TRANSCRIPTION:
            executor, backend, resources = self._bootstrap.transcription_executor()
        elif role == WorkerRole.PIPELINE:
            executor, backend, resources = self._bootstrap.pipeline_executor()
        else:
            raise RuntimeError("unsupported_entrypoint_role")
        self._resources = resources
        ServerlessRuntime(
            role,
            executor,
            backend,
        ).start()


if __name__ == "__main__":
    ServerlessEntrypoint().run()