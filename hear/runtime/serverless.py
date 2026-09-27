from __future__ import annotations

import asyncio
import importlib
from datetime import UTC, datetime

from hear.contracts.events import ExecutionEventType
from hear.contracts.jobs import AttemptEnvelope
from hear.execution.executor import JobExecutor
from hear.execution.reporter import BackendAttemptClient
from hear.health.service import RuntimeReadiness
from hear.runtime.attempt_stream import AttemptRejection, AttemptStream
from hear.runtime.roles import WorkerRole


class ServerlessRuntime:
    def __init__(
        self,
        role: WorkerRole,
        executor: JobExecutor,
        backend: BackendAttemptClient,
        provider=None,
        readiness: RuntimeReadiness | None = None,
        max_concurrent_jobs: int = 1,
    ) -> None:
        if max_concurrent_jobs < 1:
            raise ValueError("invalid_serverless_concurrency")
        self._role = role
        self._attempt_stream = AttemptStream(role, executor, backend)
        self._provider = provider
        self._readiness = readiness
        self._admission = asyncio.Semaphore(max_concurrent_jobs)

    def _runpod(self):
        if self._provider is None:
            self._provider = importlib.import_module("runpod")
        return self._provider

    async def handler(self, job):
        envelope = AttemptEnvelope.model_validate(job["input"])
        provider = self._runpod()
        remaining = (envelope.deadline - datetime.now(UTC)).total_seconds()
        if remaining <= 0:
            yield {
                "event": "attempt_deadline_exceeded",
                "job_id": envelope.job_id,
                "attempt_id": envelope.attempt_id,
            }
            return
        try:
            async with asyncio.timeout(remaining):
                await self._admission.acquire()
        except TimeoutError:
            yield {
                "event": "attempt_deadline_exceeded",
                "job_id": envelope.job_id,
                "attempt_id": envelope.attempt_id,
            }
            return
        try:
            if self._readiness is not None and not self._readiness.is_ready():
                raise RuntimeError("runtime_not_ready")
            attempt = await self._attempt_stream.prepare(envelope)
            if isinstance(attempt, AttemptRejection):
                yield {"event": attempt.event, **attempt.data}
                return
            async for event in self._attempt_stream.stream(attempt):
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
            self._admission.release()

    def start(self) -> None:
        serverless = self._runpod().serverless
        register_fitness_check = getattr(serverless, "register_fitness_check", None)
        if not callable(register_fitness_check):
            raise RuntimeError("runpod_fitness_checks_unavailable")
        register_fitness_check(self._check_readiness)
        serverless.start(
            {
                "handler": self.handler,
                "return_aggregate_stream": False,
            }
        )

    def _check_readiness(self) -> None:
        if self._readiness is None or not self._readiness.is_ready():
            raise RuntimeError("runtime_not_ready")
