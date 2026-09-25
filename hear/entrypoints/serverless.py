from hear.bootstrap import RuntimeBootstrap
from hear.runtime.roles import WorkerRole
from hear.runtime.serverless import ServerlessRuntime


class ServerlessEntrypoint:
    def __init__(self, bootstrap: RuntimeBootstrap | None = None) -> None:
        self._bootstrap = bootstrap or RuntimeBootstrap()

    def run(self) -> None:
        executor, backend, resources = self._bootstrap.transcription_executor()
        self._resources = resources
        ServerlessRuntime(
            WorkerRole.TRANSCRIPTION,
            executor,
            backend,
        ).start()


if __name__ == "__main__":
    ServerlessEntrypoint().run()