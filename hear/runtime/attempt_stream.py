from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime

from hear.contracts.events import ExecutionEvent
from hear.contracts.jobs import AttemptClaim, AttemptEnvelope, ClaimDecision
from hear.execution.executor import JobExecutor
from hear.execution.lease import AttemptLease
from hear.execution.reporter import BackendAttemptClient
from hear.runtime.roles import WorkerCapabilityRegistry, WorkerRole


@dataclass(frozen=True)
class AttemptRejection:
    event: str
    data: dict


class AttemptDeadlineExceeded(RuntimeError):
    pass


@dataclass
class PreparedAttempt:
    envelope: AttemptEnvelope
    lease: AttemptLease
    iterator: AsyncIterator[ExecutionEvent]
    closed: bool = False

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            await self.lease.close()
        finally:
            close = getattr(self.iterator, "aclose", None)
            if callable(close):
                await close()


class AttemptStream:
    def __init__(
        self,
        role: WorkerRole,
        executor: JobExecutor,
        backend: BackendAttemptClient,
    ) -> None:
        self._capability = WorkerCapabilityRegistry().get(role)
        self._executor = executor
        self._backend = backend

    async def prepare(
        self,
        envelope: AttemptEnvelope,
    ) -> PreparedAttempt | AttemptRejection:
        if datetime.now(UTC) >= envelope.deadline:
            return AttemptRejection(
                "attempt_deadline_exceeded",
                {"job_id": envelope.job_id, "attempt_id": envelope.attempt_id},
            )
        if not self._capability.accepts(envelope):
            return AttemptRejection(
                "capability_rejected",
                {"job_id": envelope.job_id, "attempt_id": envelope.attempt_id},
            )
        claim: AttemptClaim = await self._backend.claim(envelope)
        if claim.decision != ClaimDecision.EXECUTE:
            return AttemptRejection(
                "claim_rejected",
                {
                    "decision": claim.decision.value,
                    "job_id": envelope.job_id,
                    "attempt_id": envelope.attempt_id,
                },
            )
        lease = AttemptLease(self._backend, envelope, claim)
        lease.start()
        return PreparedAttempt(
            envelope=envelope,
            lease=lease,
            iterator=self._executor.stream(envelope).__aiter__(),
        )

    async def stream(
        self,
        attempt: PreparedAttempt,
    ) -> AsyncIterator[ExecutionEvent]:
        try:
            remaining = (attempt.envelope.deadline - datetime.now(UTC)).total_seconds()
            if remaining <= 0:
                raise AttemptDeadlineExceeded("attempt_deadline_exceeded")
            try:
                async with asyncio.timeout(remaining):
                    while True:
                        try:
                            yield await attempt.lease.next_event(attempt.iterator)
                        except StopAsyncIteration:
                            return
            except TimeoutError as exc:
                raise AttemptDeadlineExceeded("attempt_deadline_exceeded") from exc
        finally:
            await attempt.close()
