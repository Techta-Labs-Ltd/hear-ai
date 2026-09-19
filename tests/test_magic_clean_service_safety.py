from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from hear.deployments.magic_clean import MagicCleanDeployment
from hear.services.magic_clean.blocking import run_awaitable_to_completion
from hear.services.magic_clean.models import DEFAULT_STEM_LEVELS, StemLevels
from hear.services.magic_clean.service import (
    MagicCleanAudioEnhancer,
    _run_blocking_to_completion,
)


def test_stem_controls_are_all_omitted_or_all_explicit():
    assert MagicCleanAudioEnhancer._stem_levels(None, None, None) == DEFAULT_STEM_LEVELS
    assert MagicCleanAudioEnhancer._stem_levels(0, 100, 0) == StemLevels(0, 100, 0)
    with pytest.raises(ValueError, match="supplied together"):
        MagicCleanAudioEnhancer._stem_levels(100, None, 10)


def test_source_retention_gate_respects_intentional_component_removal():
    assert MagicCleanAudioEnhancer._preserves_all_source_components(
        DEFAULT_STEM_LEVELS
    )
    assert not MagicCleanAudioEnhancer._preserves_all_source_components(
        StemLevels(speech=0, music=100, background=0)
    )
    assert not MagicCleanAudioEnhancer._preserves_all_source_components(
        StemLevels(speech=0, music=0, background=0)
    )


class _ObservedLock:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.waiting = asyncio.Event()

    async def hold(self) -> None:
        await self._lock.acquire()

    async def __aenter__(self):
        self.waiting.set()
        await self._lock.acquire()
        return self

    async def __aexit__(self, *_args) -> None:
        self._lock.release()

    def release(self) -> None:
        self._lock.release()


async def _exercise_cancellation_before_gpu_lock_does_not_allocate_output(
    monkeypatch,
    tmp_path,
):
    output_dir = tmp_path / "standalone-output"
    allocations: list[str] = []

    def allocate_output(_purpose: str) -> str:
        output_dir.mkdir()
        allocations.append(str(output_dir))
        return str(output_dir)

    monkeypatch.setattr(
        "hear.services.magic_clean.service.magic_clean_artifact_hashes",
        lambda _path: ("source-file", "source-pcm"),
    )
    monkeypatch.setattr(
        "hear.services.magic_clean.service.hear_temp_standalone_dir",
        allocate_output,
    )
    enhancer = MagicCleanAudioEnhancer.__new__(MagicCleanAudioEnhancer)
    enhancer._loaded = True
    enhancer._gpu_lock = _ObservedLock()
    await enhancer._gpu_lock.hold()

    task = asyncio.create_task(
        enhancer.enhance(
            input_path="source.audio",
            track_id="track",
            job_id="job",
            storage=SimpleNamespace(),
        )
    )
    await asyncio.wait_for(enhancer._gpu_lock.waiting.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    enhancer._gpu_lock.release()

    assert allocations == []
    assert not output_dir.exists()


def test_cancellation_before_gpu_lock_does_not_leak_standalone_output(
    monkeypatch,
    tmp_path,
):
    asyncio.run(
        _exercise_cancellation_before_gpu_lock_does_not_allocate_output(
            monkeypatch,
            tmp_path,
        )
    )


async def _exercise_failure_before_output_creation_cleans_standalone_directory(
    monkeypatch,
    tmp_path,
):
    output_dir = tmp_path / "standalone-output"

    def allocate_output(_purpose: str) -> str:
        output_dir.mkdir()
        return str(output_dir)

    def fail_before_output(*_args, **_kwargs):
        raise RuntimeError("processing failed before output creation")

    monkeypatch.setattr(
        "hear.services.magic_clean.service.magic_clean_artifact_hashes",
        lambda _path: ("source-file", "source-pcm"),
    )
    monkeypatch.setattr(
        "hear.services.magic_clean.service.hear_temp_standalone_dir",
        allocate_output,
    )
    monkeypatch.setattr(
        "hear.services.magic_clean.service.clean_file_streaming",
        fail_before_output,
    )
    enhancer = MagicCleanAudioEnhancer.__new__(MagicCleanAudioEnhancer)
    enhancer._loaded = True
    enhancer._gpu_lock = asyncio.Lock()
    enhancer._pipeline = SimpleNamespace()
    enhancer._device = SimpleNamespace(type="cpu")

    with pytest.raises(RuntimeError, match="before output creation"):
        await enhancer.enhance(
            input_path="source.audio",
            track_id="track",
            job_id="job",
            storage=SimpleNamespace(),
        )

    assert not output_dir.exists()


def test_failure_before_output_creation_cleans_standalone_directory(
    monkeypatch,
    tmp_path,
):
    asyncio.run(
        _exercise_failure_before_output_creation_cleans_standalone_directory(
            monkeypatch,
            tmp_path,
        )
    )


async def _exercise_actor_download_cancellation_cleanup(monkeypatch):
    deployment_class = MagicCleanDeployment.func_or_class
    deployment = deployment_class.__new__(deployment_class)
    cleanup_calls: list[tuple[object, str, str]] = []

    async def cancelled_download(*_args, **_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(
        "hear.deployments.magic_clean.download_audio",
        cancelled_download,
    )
    monkeypatch.setattr(
        "hear.deployments.magic_clean.cleanup_job_temp",
        lambda db, job_id, run_id: cleanup_calls.append((db, job_id, run_id)),
    )

    with pytest.raises(asyncio.CancelledError):
        await deployment.enhance(
            audio_url="https://audio.test/source.wav",
            track_id="track",
            job_id="job",
            ai_job_id="job",
            ai_run_id="run",
        )

    assert cleanup_calls == [(None, "job", "run")]


def test_actor_cancellation_during_download_cleans_job_scope(monkeypatch):
    asyncio.run(_exercise_actor_download_cancellation_cleanup(monkeypatch))


async def _exercise_cancellation_waits_for_writer_thread():
    started = threading.Event()
    release = threading.Event()

    def writer():
        started.set()
        assert release.wait(timeout=5)
        return "written"

    task = asyncio.create_task(_run_blocking_to_completion(writer))
    await asyncio.to_thread(started.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_cancellation_does_not_abandon_writer_thread():
    asyncio.run(_exercise_cancellation_waits_for_writer_thread())


async def _exercise_blocking_worker_self_cancellation(monkeypatch):
    shield_calls = 0
    original_shield = asyncio.shield

    def guarded_shield(awaitable):
        nonlocal shield_calls
        shield_calls += 1
        if shield_calls > 1:
            raise AssertionError("completed cancelled worker was awaited again")
        return original_shield(awaitable)

    monkeypatch.setattr("hear.core.blocking.asyncio.shield", guarded_shield)

    def self_cancelling_writer():
        raise asyncio.CancelledError("worker stopped")

    with pytest.raises(asyncio.CancelledError, match="worker stopped"):
        await _run_blocking_to_completion(self_cancelling_writer)

    assert shield_calls == 1


def test_blocking_worker_self_cancellation_does_not_spin_forever(monkeypatch):
    asyncio.run(_exercise_blocking_worker_self_cancellation(monkeypatch))


async def _exercise_cancellation_waits_for_remote_writer():
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def writer():
        started.set()
        await release.wait()
        finished.set()
        return "written"

    task = asyncio.create_task(run_awaitable_to_completion(writer()))
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()


def test_cancellation_does_not_abandon_remote_writer():
    asyncio.run(_exercise_cancellation_waits_for_remote_writer())


async def _exercise_cancellation_signals_remote_worker():
    started = asyncio.Event()
    stop_requested = asyncio.Event()
    cancel_calls = 0

    async def worker():
        started.set()
        await stop_requested.wait()
        raise RuntimeError("remote worker stopped")

    def cancel_worker():
        nonlocal cancel_calls
        cancel_calls += 1
        stop_requested.set()

    task = asyncio.create_task(
        run_awaitable_to_completion(worker(), on_cancel=cancel_worker)
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert cancel_calls == 1


def test_cancellation_signals_and_observes_remote_worker():
    asyncio.run(_exercise_cancellation_signals_remote_worker())


async def _exercise_remote_self_cancellation_propagates():
    async def self_cancelling_writer():
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(
            run_awaitable_to_completion(self_cancelling_writer()),
            timeout=1,
        )


def test_remote_self_cancellation_does_not_spin_forever():
    asyncio.run(_exercise_remote_self_cancellation_propagates())
