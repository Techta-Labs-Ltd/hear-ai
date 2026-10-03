"""Pod and Serverless must emit byte-identical canonical events and backend outcomes."""

import json
from types import SimpleNamespace

import pytest

from hear.contracts.events import ExecutionEvent, ExecutionEventType
from hear.contracts.jobs import AttemptClaim, ClaimDecision, JobType
from hear.contracts.outcomes import ExecutionOutcome
from hear.execution.executor import JobExecutor
from hear.runtime.pod import PodRuntime
from hear.runtime.roles import WorkerRole
from hear.runtime.serverless import ServerlessRuntime
from tests.test_pod_runtime_v2 import FakeReadiness, envelope


class RecordingBackend:
    def __init__(self):
        self.events = []
        self.outcomes = []

    async def claim(self, _envelope):
        return AttemptClaim(decision=ClaimDecision.EXECUTE)

    async def heartbeat(self, _envelope, _sequence):
        return None

    async def event(self, _envelope, event):
        self.events.append(event.model_dump(mode="json"))

    async def outcome(self, _envelope, outcome):
        self.outcomes.append(outcome.model_dump(mode="json"))


class DeterministicWorkflow:
    """Same event ids for both transports so payloads can be compared verbatim."""

    async def stream(self, request):
        base = dict(
            job_id=request.job_id,
            attempt_id=request.attempt_id,
            track_id=request.track_id,
            job_type=request.job_type,
            source_revision=request.source.revision,
        )
        yield ExecutionEvent(
            event_id="e1",
            sequence=1,
            event=ExecutionEventType.STARTED,
            stage="preparing",
            progress_pct=0,
            **base,
        )
        yield ExecutionEvent(
            event_id="e2",
            sequence=2,
            event=ExecutionEventType.PROGRESS,
            stage="transcribing",
            progress_pct=40,
            **base,
        )
        outcome = ExecutionOutcome(status="completed", result={"language": "en"}, **base)
        yield ExecutionEvent(
            event_id="e3",
            sequence=3,
            event=ExecutionEventType.OUTCOME,
            stage="completed",
            progress_pct=100,
            data={"outcome": outcome.model_dump(mode="json")},
            **base,
        )


class ReplyExchange:
    def __init__(self):
        self.payloads = []

    async def publish(self, message, **_kwargs):
        self.payloads.append(json.loads(message.body))


async def run_pod(request):
    backend = RecordingBackend()
    runtime = PodRuntime(
        FakeReadiness(),
        WorkerRole.TRANSCRIPTION,
        JobExecutor({JobType.TRANSCRIPTION: DeterministicWorkflow()}),
        backend,
        api_key="pod-key",
    )
    await runtime.start()
    exchange = ReplyExchange()
    runtime._rabbitmq_channel = SimpleNamespace(default_exchange=exchange, is_closed=False)
    runtime._rabbitmq_provider = SimpleNamespace(Message=lambda **kw: SimpleNamespace(**kw))
    acknowledged = []

    async def ack():
        acknowledged.append(True)

    message = SimpleNamespace(body=request.model_dump_json().encode(), reply_to="preview", ack=ack)
    await runtime._on_rabbitmq_message(message)
    assert acknowledged == [True]
    events = [item["data"] for item in exchange.payloads if item["kind"] == "event"]
    assert exchange.payloads[-1] == {"kind": "end"}
    return events, backend


async def run_serverless(request):
    backend = RecordingBackend()
    runtime = ServerlessRuntime(
        WorkerRole.TRANSCRIPTION,
        JobExecutor({JobType.TRANSCRIPTION: DeterministicWorkflow()}),
        backend,
        SimpleNamespace(serverless=SimpleNamespace(progress_update=lambda *_a: None)),
    )
    job = {"id": "rp-1", "input": request.model_dump(mode="json")}
    events = [event async for event in runtime.handler(job)]
    return events, backend


@pytest.mark.anyio
async def test_pod_and_serverless_emit_identical_events_and_backend_outcomes():
    request = envelope()
    pod_events, pod_backend = await run_pod(request)
    serverless_events, serverless_backend = await run_serverless(request)

    assert pod_events == serverless_events
    assert [item["event"] for item in pod_events] == ["started", "progress", "outcome"]
    assert pod_backend.outcomes == serverless_backend.outcomes
    assert pod_backend.events == serverless_backend.events
    assert len(pod_backend.outcomes) == 1
    assert pod_backend.outcomes[0]["status"] == "completed"
    # Transport output is a preview; the backend callback carries the identical outcome.
    assert pod_events[-1]["data"]["outcome"] == pod_backend.outcomes[0]
