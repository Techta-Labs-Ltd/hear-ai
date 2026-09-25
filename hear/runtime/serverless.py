from collections.abc import AsyncIterator

from hear.contracts.events import ExecutionEvent
from hear.contracts.jobs import AttemptEnvelope
from hear.execution.executor import JobExecutor


class ServerlessRuntime:
    def __init__(self, executor: JobExecutor) -> None:
        self._executor = executor

    async def stream(self, payload: dict) -> AsyncIterator[dict]:
        envelope = AttemptEnvelope.model_validate(payload)
        async for event in self._executor.stream(envelope):
            yield event.model_dump(mode="json")
