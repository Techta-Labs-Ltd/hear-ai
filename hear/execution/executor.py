from collections.abc import AsyncIterator
from typing import Protocol

from hear.contracts.events import ExecutionEvent, ExecutionEventType
from hear.contracts.jobs import AttemptEnvelope, JobType
from hear.runtime.roles import RoleCapabilities

from .context import ExecutionContext


class JobWorkflow(Protocol):
    async def execute(self, context: ExecutionContext) -> AsyncIterator[ExecutionEvent]: ...


class JobExecutor:
    def __init__(
        self,
        capabilities: RoleCapabilities,
        workflows: dict[JobType, JobWorkflow],
    ) -> None:
        self._capabilities = capabilities
        self._workflows = dict(workflows)

    async def stream(self, envelope: AttemptEnvelope) -> AsyncIterator[ExecutionEvent]:
        if not self._capabilities.accepts(envelope.job_type, envelope.magic_clean_profile):
            raise RuntimeError("worker_capability_mismatch")
        workflow = self._workflows.get(envelope.job_type)
        if workflow is None:
            raise RuntimeError("workflow_not_configured")
        context = ExecutionContext(envelope)
        yield context.event(ExecutionEventType.STARTED, progress_pct=0.0)
        async for event in workflow.execute(context):
            yield event
