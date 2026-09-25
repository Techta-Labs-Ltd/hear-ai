from datetime import UTC, datetime
from uuid import uuid4

from hear.contracts.events import ExecutionEvent, ExecutionEventType
from hear.contracts.jobs import AttemptEnvelope


class ExecutionContext:
    def __init__(self, envelope: AttemptEnvelope) -> None:
        self.envelope = envelope
        self._sequence = 0

    def event(
        self,
        event: ExecutionEventType,
        *,
        stage: str | None = None,
        progress_pct: float | None = None,
        message: str | None = None,
        metrics: dict | None = None,
    ) -> ExecutionEvent:
        self._sequence += 1
        return ExecutionEvent(
            event_id=str(uuid4()),
            job_id=self.envelope.job_id,
            run_id=self.envelope.run_id,
            attempt_id=self.envelope.attempt_id,
            track_id=self.envelope.track_id,
            job_type=self.envelope.job_type,
            source_revision=self.envelope.source.revision,
            sequence=self._sequence,
            event=event,
            stage=stage,
            progress_pct=progress_pct,
            message=message,
            metrics=metrics or {},
            created_at=datetime.now(UTC),
        )
