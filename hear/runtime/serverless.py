from __future__ import annotations

import runpod

from hear.contracts.jobs import AttemptEnvelope
from hear.execution.executor import JobExecutor
from hear.execution.reporter import BackendAttemptClient


class ServerlessRuntime:
    def __init__(self, executor: JobExecutor, backend: BackendAttemptClient) -> None:
        self._executor = executor
        self._backend = backend

    async def handle(self, job: dict):
        request = AttemptEnvelope.model_validate(job.get("input") or {})
        decision = await self._backend.claim(request)
        if decision != "execute":
            yield {"event": "claim", "decision": decision, "attempt_id": request.attempt_id}
            return
        async for event in self._executor.stream(request):
            payload = event.model_dump(mode="json")
            runpod.serverless.progress_update(job, {
                "stage": event.stage,
                "progress_pct": event.progress_pct,
                "sequence": event.sequence,
            })
            yield payload

    async def close(self) -> None:
        await self._backend.close()
