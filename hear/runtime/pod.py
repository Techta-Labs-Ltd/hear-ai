import httpx

from hear.contracts.jobs import AttemptEnvelope
from hear.execution.executor import JobExecutor
from hear.execution.reporter import BackendReporter


class PodRuntime:
    def __init__(
        self,
        executor: JobExecutor,
        client: httpx.AsyncClient,
    ) -> None:
        self._executor = executor
        self._client = client

    async def execute(self, envelope: AttemptEnvelope) -> None:
        reporter = BackendReporter(envelope, self._client)
        async for event in self._executor.stream(envelope):
            await reporter.publish_event(event)
