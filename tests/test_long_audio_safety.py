import asyncio
import json
import os
import shutil
from collections import namedtuple
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from hear.contracts.events import ExecutionEvent, ExecutionEventType
from hear.contracts.jobs import AttemptEnvelope, JobType
from hear.execution.executor import FailurePolicy, JobExecutor
from hear.runtime.cleaner.scratch_ledger import HostScratchLedger
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode

GIB = 1024**3
Usage = namedtuple("Usage", "total used free")


def fake_disk(monkeypatch, free: int) -> None:
    monkeypatch.setattr(shutil, "disk_usage", lambda _path: Usage(100 * GIB, 0, free))


def workspace(root, name: str):
    path = root / name
    path.mkdir()
    return path


def test_ledger_admits_jobs_until_the_disk_is_promised(tmp_path, monkeypatch):
    fake_disk(monkeypatch, 20 * GIB)
    ledger = HostScratchLedger(tmp_path, min_free_bytes=1 * GIB)
    ledger.reserve(workspace(tmp_path, "a"), 12 * GIB)
    with pytest.raises(CleanExecutionError) as busy:
        ledger.reserve(workspace(tmp_path, "b"), 12 * GIB)
    assert busy.value.code == ErrorCode.RESOURCE_EXHAUSTED and busy.value.retryable
    ledger.reserve(tmp_path / "b", 7 * GIB)


def test_ledger_queues_a_blocked_job_until_space_frees(tmp_path, monkeypatch):
    fake_disk(monkeypatch, 20 * GIB)
    ledger = HostScratchLedger(tmp_path, min_free_bytes=1 * GIB)
    first = workspace(tmp_path, "first")
    ledger.reserve(first, 12 * GIB)
    polls = []

    def wait():
        polls.append(1)
        if len(polls) == 2:
            first.rmdir()  # the first job finished and cleaned up

    ledger.reserve(workspace(tmp_path, "second"), 12 * GIB, wait=wait, poll_seconds=0)
    assert len(polls) == 2


def test_ledger_wait_gives_up_when_the_attempt_expires(tmp_path, monkeypatch):
    fake_disk(monkeypatch, 20 * GIB)
    ledger = HostScratchLedger(tmp_path, min_free_bytes=1 * GIB)
    ledger.reserve(workspace(tmp_path, "first"), 12 * GIB)

    def expired():
        raise CleanExecutionError(ErrorCode.DEADLINE_EXCEEDED, "attempt deadline exceeded")

    with pytest.raises(CleanExecutionError) as stopped:
        ledger.reserve(workspace(tmp_path, "second"), 12 * GIB, wait=expired, poll_seconds=0)
    assert stopped.value.code == ErrorCode.DEADLINE_EXCEEDED


def test_disk_written_by_other_jobs_makes_a_job_wait_not_fail(tmp_path, monkeypatch):
    # 6 GiB free now because another job has written 10 GiB it will release when done.
    fake_disk(monkeypatch, 6 * GIB)
    ledger = HostScratchLedger(tmp_path, min_free_bytes=1 * GIB)
    monkeypatch.setattr(
        HostScratchLedger,
        "_written",
        staticmethod(lambda path: 10 * GIB if path.name == "other" else 0),
    )
    ledger.reserve(workspace(tmp_path, "other"), 10 * GIB)
    with pytest.raises(CleanExecutionError) as busy:
        ledger.reserve(workspace(tmp_path, "mine"), 12 * GIB)
    assert busy.value.retryable


def test_ledger_rejects_a_job_larger_than_the_idle_host_as_final(tmp_path, monkeypatch):
    fake_disk(monkeypatch, 10 * GIB)
    ledger = HostScratchLedger(tmp_path, min_free_bytes=1 * GIB)
    with pytest.raises(CleanExecutionError) as too_big:
        ledger.reserve(workspace(tmp_path, "a"), 25 * GIB)
    assert not too_big.value.retryable


def test_ledger_releases_finished_and_dead_workspaces(tmp_path, monkeypatch):
    fake_disk(monkeypatch, 20 * GIB)
    ledger = HostScratchLedger(tmp_path, min_free_bytes=0)
    finished = workspace(tmp_path, "finished")
    ledger.reserve(finished, 15 * GIB)
    finished.rmdir()
    ledger.reserve(workspace(tmp_path, "next"), 15 * GIB)
    state = json.loads((tmp_path / ".scratch-ledger.json").read_text())
    assert list(state) == [str((tmp_path / "next").resolve())]
    assert state[str((tmp_path / "next").resolve())]["pid"] == os.getpid()


@pytest.mark.parametrize(
    ("error", "final"),
    [
        (CleanExecutionError(ErrorCode.RESOURCE_EXHAUSTED, "audio duration exceeds limit"), "resource_exhausted"),
        (CleanExecutionError(ErrorCode.INVALID_AUDIO, "bad"), "invalid_audio"),
        (CleanExecutionError(ErrorCode.RESOURCE_EXHAUSTED, "busy", retryable=True), None),
        (CleanExecutionError(ErrorCode.PROCESS_FAILED, "gpu", worker_restart_required=True), None),
        (CleanExecutionError(ErrorCode.STORAGE_FAILED, "b2"), None),
        (ValueError("reconstruction_source_too_large"), "invalid_request"),
        (RuntimeError("unknown"), None),
    ],
)
def test_failure_policy_separates_final_from_retryable(error, final):
    assert FailurePolicy.final_error_code(error) == final


@pytest.mark.parametrize(("status", "final"), [(404, "source_unavailable"), (503, None), (429, None)])
def test_source_download_status_decides_retry(status, final):
    request = httpx.Request("GET", "https://cdn.example/source.mp3")
    error = httpx.HTTPStatusError("x", request=request, response=httpx.Response(status, request=request))
    assert FailurePolicy.final_error_code(error) == final


def envelope() -> AttemptEnvelope:
    expiry = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    return AttemptEnvelope.model_validate(
        {
            "job_id": "job",
            "run_id": "run",
            "attempt_id": "attempt",
            "job_type": "reconstruction",
            "operation": "rebuild",
            "track_id": "track",
            "user_id": "user",
            "source": {"url": "https://cdn.example/source.mp3", "revision": 3},
            "storage": {
                "endpoint_url": "https://s3.example/",
                "bucket_name": "bucket",
                "key_id": "k",
                "application_key": "s",
                "folder_prefix": "creators/x/audio/jobs/job/",
                "public_base_url": "https://cdn.example/",
                "expires_at": expiry,
            },
            "options": {"edited_transcript": "x", "same_speaker": False},
            "artifact_prefix": "creators/x/audio/jobs/job/attempt",
            "deadline": expiry,
            "reporting_grant": "grant",
            "backend_base_url": "https://api.example/api/v1",
            "backend_id": "backend-a",
        }
    )


class FailingWorkflow:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def stream(self, value: AttemptEnvelope):
        yield ExecutionEvent(
            event_id="e1",
            job_id=value.job_id,
            attempt_id=value.attempt_id,
            track_id=value.track_id,
            job_type=value.job_type,
            backend_id=value.backend_id,
            source_revision=value.source.revision,
            sequence=1,
            event=ExecutionEventType.STARTED,
        )
        raise self.error


async def collect(executor: JobExecutor, value: AttemptEnvelope) -> list[ExecutionEvent]:
    return [event async for event in executor.stream(value)]


def test_final_error_is_reported_to_the_backend_as_a_failed_outcome():
    value = envelope()
    executor = JobExecutor(
        {JobType.RECONSTRUCTION: FailingWorkflow(CleanExecutionError(ErrorCode.RESOURCE_EXHAUSTED, "audio duration exceeds limit"))}
    )
    events = asyncio.run(collect(executor, value))
    outcome = events[-1].data["outcome"]
    assert events[-1].event == ExecutionEventType.OUTCOME and events[-1].sequence == 2
    assert outcome["status"] == "failed" and outcome["error_code"] == "resource_exhausted"
    assert outcome["backend_id"] == "backend-a" and outcome["source_revision"] == 3


def test_transient_error_still_propagates_for_a_retry():
    executor = JobExecutor({JobType.RECONSTRUCTION: FailingWorkflow(RuntimeError("cuda hiccup"))})
    with pytest.raises(RuntimeError, match="cuda hiccup"):
        asyncio.run(collect(executor, envelope()))


def test_gpu_memory_floor_rejects_small_cards(monkeypatch):
    import types

    import torch

    from hear.bootstrap import RuntimeBootstrap

    def card(gigabytes):
        return types.SimpleNamespace(total_memory=int(gigabytes * 1e9))

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: card(19.6))
    assert not RuntimeBootstrap._gpu_has_capacity(22)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: card(24.0))
    assert RuntimeBootstrap._gpu_has_capacity(22)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert not RuntimeBootstrap._gpu_has_capacity(22)
