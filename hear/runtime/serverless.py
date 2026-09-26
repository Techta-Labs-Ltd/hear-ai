from __future__ import annotations

import importlib

from hear.contracts.events import ExecutionEventType
from hear.contracts.jobs import AttemptEnvelope, ClaimDecision
from hear.execution.executor import JobExecutor
from hear.execution.lease import AttemptLease
from hear.execution.reporter import BackendAttemptClient
from hear.runtime.roles import WorkerCapabilityRegistry, WorkerRole


class ServerlessRuntime:
    def __init__(
        self,
        role: WorkerRole,
        executor: JobExecutor,
        backend: BackendAttemptClient,
        provider=None,
    ) -> None:
        self._role = role
        self._capability = WorkerCapabilityRegistry().get(role)
        self._executor = executor
        self._backend = backend
        self._provider = provider

    def _runpod(self):
        if self._provider is None:
            self._provider = importlib.import_module("runpod")
        return self._provider

    async def handler(self, job):
        envelope = AttemptEnvelope.model_validate(job["input"])
        if not self._capability.accepts(envelope):
            yield {
                "event": "capability_rejected",
                "job_id": envelope.job_id,
                "attempt_id": envelope.attempt_id,
            }
            return
        claim = await self._backend.claim(envelope)
        if claim.decision != ClaimDecision.EXECUTE:
            yield {
                "event": "claim_rejected",
                "decision": claim.decision.value,
                "job_id": envelope.job_id,
                "attempt_id": envelope.attempt_id,
            }
            return
        provider = self._runpod()
        lease = AttemptLease(self._backend, envelope, claim)
        lease.start()
        iterator = self._executor.stream(envelope).__aiter__()
        try:
            while True:
                try:
                    event = await lease.next_event(iterator)
                except StopAsyncIteration:
                    break
                if event.event in {
                    ExecutionEventType.STAGE,
                    ExecutionEventType.PROGRESS,
                    ExecutionEventType.ARTIFACT_PREPARED,
                    ExecutionEventType.OUTCOME,
                }:
                    provider.serverless.progress_update(
                        job,
                        (
                            f"{event.stage or event.event.value}:"
                            f"{event.progress_pct if event.progress_pct is not None else ''}"
                        ),
                    )
                yield event.model_dump(mode="json")
        finally:
            await lease.close()
            close = getattr(iterator, "aclose", None)
            if callable(close):
                await close()

    def start(self) -> None:
        self._runpod().serverless.start(
            {
                "handler": self.handler,
                "return_aggregate_stream": True,
            }
        )
