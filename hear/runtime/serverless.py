from __future__ import annotations

import asyncio
import importlib
import logging
from datetime import UTC, datetime

import httpx

from hear.contracts.events import ExecutionEventType
from hear.contracts.jobs import AttemptEnvelope
from hear.contracts.outcomes import ExecutionOutcome
from hear.execution.executor import FailureSummary, JobExecutor, WorkerFailure
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
        if role == WorkerRole.RECONSTRUCTION and max_concurrent_jobs != 1:
            raise ValueError("fish_reconstruction_requires_one_job_per_worker")
        self._backend = backend
        self._role = role
        self._attempt_stream = AttemptStream(role, executor, backend)
        self._provider = provider
        self._readiness = readiness
        self._max_concurrent_jobs = max_concurrent_jobs
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
        prepared = None
        try:
            if self._readiness is not None and not self._readiness.is_ready():
                raise RuntimeError("runtime_not_ready")
            try:
                attempt = await self._attempt_stream.prepare(envelope)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in BackendAttemptClient.REJECTED_STATUSES:
                    raise
                # Same verdict the Pod consumer publishes: the backend disowned the attempt.
                yield {
                    "event": "backend_attempt_rejected",
                    "job_id": envelope.job_id,
                    "attempt_id": envelope.attempt_id,
                    "http_status": exc.response.status_code,
                }
                return
            if isinstance(attempt, AttemptRejection):
                yield {"event": attempt.event, **attempt.data}
                return
            prepared = attempt
            async for event in self._attempt_stream.stream(attempt):
                # Returning RunPod output alone does not persist the result in Hear.
                if event.event == ExecutionEventType.OUTCOME:
                    outcome = ExecutionOutcome.model_validate(event.data.get("outcome"))
                    await self._deliver(self._backend.outcome, envelope, outcome)
                else:
                    report_event = getattr(self._backend, "event", None)
                    if callable(report_event):
                        await self._deliver(report_event, envelope, event)
                if event.event in {
                    ExecutionEventType.STAGE,
                    ExecutionEventType.PROGRESS,
                    ExecutionEventType.ARTIFACT_PREPARED,
                    ExecutionEventType.OUTCOME,
                }:
                    try:
                        provider.serverless.progress_update(
                            job,
                            (
                                f"{event.stage or event.event.value}:"
                                f"{event.progress_pct if event.progress_pct is not None else ''}"
                            ),
                        )
                    except (OSError, httpx.HTTPError):
                        logging.getLogger(__name__).warning("Optional provider progress update failed")
                yield event.model_dump(mode="json")
        except Exception as exc:
            # RunPod keeps "handler: <message>" plus format_exc(); a chained process-pool
            # traceback would push the real error past what the backend stores.
            logging.getLogger(__name__).exception(
                "serverless_attempt_failed job_id=%s attempt_id=%s",
                envelope.job_id,
                envelope.attempt_id,
            )
            raise WorkerFailure(FailureSummary.describe(exc)) from None
        finally:
            try:
                if prepared is not None:
                    await prepared.close()
            finally:
                self._admission.release()

    @staticmethod
    async def _deliver(callback, *args) -> None:

        for index, delay in enumerate((0, 1, 2, 4)):
            if delay:
                await asyncio.sleep(delay)
            try:
                await callback(*args)
                return
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500 and exc.response.status_code != 429:
                    raise
                if index == 3:
                    raise
            except (httpx.TransportError, OSError):
                if index == 3:
                    raise

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
                "concurrency_modifier": lambda current: self._max_concurrent_jobs,
            }
        )

    def _check_readiness(self) -> None:
        if self._readiness is None or not self._readiness.is_ready():
            raise RuntimeError("runtime_not_ready")
