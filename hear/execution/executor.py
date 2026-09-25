from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from hear.contracts.events import ExecutionEvent
from hear.contracts.jobs import AttemptEnvelope, JobType


class ExecutionWorkflow(Protocol):
    async def stream(self, envelope: AttemptEnvelope) -> AsyncIterator[ExecutionEvent]: ...


class JobExecutor:
    def __init__(self, workflows: dict[JobType, ExecutionWorkflow]) -> None:
        self._workflows = dict(workflows)

    async def stream(self, envelope: AttemptEnvelope) -> AsyncIterator[ExecutionEvent]:
        workflow = self._workflows.get(envelope.job_type)
        if workflow is None:
            raise RuntimeError(f"workflow_unavailable:{envelope.job_type.value}")
        async for event in workflow.stream(envelope):
            yield event