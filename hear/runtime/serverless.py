from __future__ import annotations

import runpod

from hear.contracts.events import ExecutionEventType
from hear.contracts.jobs import AttemptEnvelope, ClaimDecision
from hear.execution.executor import JobExecutor
from hear.execution.reporter import BackendAttemptClient
from hear.runtime.roles import WorkerCapabilityRegistry, WorkerRole


class ServerlessRuntime:
    def __init__(
        self,
        role: WorkerRole,
        executor: JobExecutor,
        backend: BackendAttemptClient,
    ) -> None:
        self._role = role
        self._capability = WorkerCapabilityRegistry().get(role)
        self._executor = executor
        self._backend = backend

    async def handler(self, job):
        envelope = AttemptEnvelope.model_validate(job["input"])
        if not self._capability.accepts(envelope):
            yield {
                "event": "capability_rejected",
                "job_id": envelope.job_id,
                "attempt_id": envelope.attempt_id,
            }
            return
        decision = await self._backend.claim(envelope)
        if decision != ClaimDecision.EXECUTE:
            yield {
                "event": "claim_rejected",
                "decision": decision.value,
                "job_id": envelope.job_id,
                "attempt_id": envelope.attempt_id,
            }
            return
        async for event in self._executor.stream(envelope):
            if event.event in {
                ExecutionEventType.STAGE,
                ExecutionEventType.PROGRESS,
                ExecutionEventType.ARTIFACT_PREPARED,
                ExecutionEventType.OUTCOME,
            }:
                runpod.serverless.progress_update(
                    job,
                    (
                        f"{event.stage or event.event.value}:"
                        f"{event.progress_pct if event.progress_pct is not None else ''}"
                    ),
                )
            yield event.model_dump(mode="json")

    def start(self) -> None:
        runpod.serverless.start(
            {
                "handler": self.handler,
                "return_aggregate_stream": True,
            }
        )