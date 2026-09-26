import json
from datetime import UTC, datetime, timedelta

import pytest

from hear.contracts.events import ExecutionEvent, ExecutionEventType
from hear.contracts.jobs import (
    AttemptClaim,
    AttemptEnvelope,
    ClaimDecision,
    JobType,
    WorkerIdentity,
)
from hear.contracts.outcomes import ExecutionOutcome
from hear.queue.rabbitmq import RabbitMQConsumer
from hear.runtime.roles import WorkerRole


class FakeMessage:
    def __init__(self, payload):
        self.body = json.dumps(payload).encode()
        self.acked = 0
        self.rejected = 0
        self.nacked = 0

    async def ack(self):
        self.acked += 1

    async def reject(self, requeue=False):
        self.rejected += 1

    async def nack(self, requeue=False):
        self.nacked += 1


class FakeBackend:
    def __init__(self):
        self.outcomes = []

    async def claim(self, envelope):
        return AttemptClaim(decision=ClaimDecision.EXECUTE)

    async def heartbeat(self, envelope, sequence):
        return None

    async def event(self, envelope, event):
        raise RuntimeError("progress_receiver_unavailable")

    async def outcome(self, envelope, outcome):
        self.outcomes.append(outcome)

    async def close(self):
        return None


class FakeExecutor:
    async def stream(self, envelope):
        yield ExecutionEvent(
            event_id="event-1",
            job_id=envelope.job_id,
            attempt_id=envelope.attempt_id,
            track_id=envelope.track_id,
            job_type=envelope.job_type,
            source_revision=envelope.source.revision,
            sequence=1,
            event=ExecutionEventType.PROGRESS,
            stage="transcribing",
            progress_pct=50,
        )
        outcome = ExecutionOutcome(
            job_id=envelope.job_id,
            attempt_id=envelope.attempt_id,
            track_id=envelope.track_id,
            job_type=envelope.job_type,
            source_revision=envelope.source.revision,
            status="completed",
        )
        yield ExecutionEvent(
            event_id="event-2",
            job_id=envelope.job_id,
            attempt_id=envelope.attempt_id,
            track_id=envelope.track_id,
            job_type=envelope.job_type,
            source_revision=envelope.source.revision,
            sequence=2,
            event=ExecutionEventType.OUTCOME,
            stage="completed",
            progress_pct=100,
            data={"outcome": outcome.model_dump(mode="json")},
        )


def payload():
    envelope = AttemptEnvelope(
        job_id="job-1",
        run_id="run-1",
        attempt_id="attempt-1",
        job_type=JobType.TRANSCRIPTION,
        track_id="track-1",
        user_id="user-1",
        source={"url": "https://example.com/a.mp3", "revision": 1},
        storage={
            "endpoint_url": "https://s3.example.com",
            "bucket_name": "bucket",
            "key_id": "key",
            "application_key": "secret",
            "folder_prefix": "users/user-1/",
            "public_base_url": "https://cdn.example.com/media",
            "expires_at": datetime.now(UTC) + timedelta(hours=1),
        },
        artifact_prefix="users/user-1/jobs/job-1/attempt-1",
        deadline=datetime.now(UTC) + timedelta(minutes=30),
        reporting_grant="grant",
        backend_base_url="https://api.example.com",
    )
    return json.loads(envelope.model_dump_json())


@pytest.mark.anyio
async def test_progress_delivery_failure_does_not_fail_execution():
    backend = FakeBackend()
    consumer = RabbitMQConsumer(
        "amqp://unused",
        WorkerRole.TRANSCRIPTION,
        FakeExecutor(),
        backend,
        WorkerIdentity(
            worker_id="worker-1",
            generation="generation-1",
            image_revision="image-1",
            engine_revision="engine-1",
        ),
    )
    message = FakeMessage(payload())
    await consumer._on_message(message)
    assert message.acked == 1
    assert len(backend.outcomes) == 1
    assert backend.outcomes[0].status == "completed"


class BlockingExecutor:
    def __init__(self):
        import asyncio

        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def stream(self, envelope):
        self.started.set()
        await self.release.wait()
        outcome = ExecutionOutcome(
            job_id=envelope.job_id,
            attempt_id=envelope.attempt_id,
            track_id=envelope.track_id,
            job_type=envelope.job_type,
            source_revision=envelope.source.revision,
            status="completed",
        )
        yield ExecutionEvent(
            event_id="event-final",
            job_id=envelope.job_id,
            attempt_id=envelope.attempt_id,
            track_id=envelope.track_id,
            job_type=envelope.job_type,
            source_revision=envelope.source.revision,
            sequence=1,
            event=ExecutionEventType.OUTCOME,
            stage="completed",
            progress_pct=100,
            data={"outcome": outcome.model_dump(mode="json")},
        )


@pytest.mark.anyio
async def test_drain_waits_for_active_attempt():
    import asyncio

    backend = FakeBackend()
    executor = BlockingExecutor()
    consumer = RabbitMQConsumer(
        "amqp://unused",
        WorkerRole.TRANSCRIPTION,
        executor,
        backend,
        WorkerIdentity(
            worker_id="worker-1",
            generation="generation-1",
            image_revision="image-1",
            engine_revision="engine-1",
        ),
    )
    message = FakeMessage(payload())
    execution = asyncio.create_task(consumer._on_message(message))
    await executor.started.wait()
    draining = asyncio.create_task(consumer.drain())
    await asyncio.sleep(0)
    assert draining.done() is False
    executor.release.set()
    await execution
    await draining
    assert backend.outcomes[-1].status == "completed"


class OutcomeFailingBackend(FakeBackend):
    async def outcome(self, envelope, outcome):
        raise RuntimeError("outcome_receiver_unavailable")


@pytest.mark.anyio
async def test_outcome_delivery_failure_is_not_reclassified_as_inference_failure():
    backend = OutcomeFailingBackend()
    consumer = RabbitMQConsumer(
        "amqp://unused",
        WorkerRole.TRANSCRIPTION,
        FakeExecutor(),
        backend,
        WorkerIdentity(
            worker_id="worker-1",
            generation="generation-1",
            image_revision="image-1",
            engine_revision="engine-1",
        ),
    )
    message = FakeMessage(payload())
    await consumer._on_message(message)
    assert message.acked == 1
    assert backend.outcomes == []
