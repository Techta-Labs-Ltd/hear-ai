import asyncio
from types import SimpleNamespace

import pytest

from hear.orchestrator import Orchestrator

OrchestratorClass = Orchestrator.func_or_class


class SnapshotReader:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.state = {"event": "job_snapshot", "job_id": "job", "status": "running"}

    def __call__(self, job_id):
        assert self.orchestrator._event_queues[job_id]
        return self.state.copy()


@pytest.fixture
def subject():
    orchestrator = object.__new__(OrchestratorClass)
    orchestrator._event_queues = {}
    reader = SnapshotReader(orchestrator)
    orchestrator._subscription_snapshot = reader
    return SimpleNamespace(orchestrator=orchestrator, reader=reader)


@pytest.mark.anyio
async def test_each_subscriber_receives_ordered_terminal_events(subject):
    orchestrator = subject.orchestrator
    streams = [orchestrator.subscribe("job"), orchestrator.subscribe("job")]
    for stream in streams:
        assert (await anext(stream))["event"] == "job_snapshot"
    events = [
        {"event": "stage_changed", "job_id": "job", "current_stage": "transcribing"},
        {"event": "job_completed", "job_id": "job", "status": "completed"},
    ]
    for event in events:
        orchestrator._push_event("job", event)
    for stream in streams:
        assert [event async for event in stream] == events
    assert orchestrator._event_queues == {}


@pytest.mark.anyio
async def test_overflow_signals_reconnect_and_recovers_terminal_snapshot(subject):
    orchestrator = subject.orchestrator
    stream = orchestrator.subscribe("job")
    await anext(stream)
    for ordinal in range(257):
        orchestrator._push_event("job", {"event": "stage_changed", "ordinal": ordinal})
    subject.reader.state = {"event": "job_completed", "job_id": "job", "status": "completed"}
    events = [event async for event in stream]
    assert [event["event"] for event in events] == ["stream_reset", "job_completed"]
    assert events[0]["error"] == "subscriber_overflow_reconnect_required"
    assert orchestrator._event_queues == {}


@pytest.mark.anyio
async def test_disconnecting_one_subscriber_preserves_another(subject):
    orchestrator = subject.orchestrator
    first = orchestrator.subscribe("job")
    second = orchestrator.subscribe("job")
    await anext(first)
    await anext(second)
    await first.aclose()
    assert len(orchestrator._event_queues["job"]) == 1
    orchestrator._push_event("job", {"event": "job_failed", "job_id": "job"})
    assert (await anext(second))["event"] == "job_failed"
    await second.aclose()
    assert orchestrator._event_queues == {}


@pytest.mark.anyio
async def test_waiting_subscriber_cancellation_releases_queue(subject):
    orchestrator = subject.orchestrator
    stream = orchestrator.subscribe("job")
    await anext(stream)
    waiter = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert orchestrator._event_queues == {}
