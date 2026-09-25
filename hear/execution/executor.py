from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from hear.contracts.events import ExecutionEvent
from hear.contracts.jobs import AttemptEnvelope


class Workflow(Protocol):
    async def stream(self, request: AttemptEnvelope) -> AsyncIterator[ExecutionEvent]: ...


class JobExecutor:
    def __init__(self, workflows: dict[str, Workflow]) -> None:
        self._workflows = dict(workflows)

    async def stream(self, request: AttemptEnvelope) -> AsyncIterator[ExecutionEvent]:
        workflow = self._workflows.get(request.job_type)
        if workflow is None:
            raise ValueError(f"unsupported_job_type:{request.job_type}")
        async for event in workflow.stream(request):
            yield event
