import hashlib
import os
import sys
import threading
import time

import numpy as np
import pytest
import soundfile as sf

from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.runtime.cleaner.subprocesses import CancellableProcessRunner
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode, SourceIdentity
from hear.services.magic_clean.inspection import SourceInspector


def inspect_fixture(tmp_path, samples, rate=48000, *, expected_frames=None):
    path = tmp_path / "source.wav"
    sf.write(path, samples, rate, subtype="FLOAT")
    content = path.read_bytes()
    source = SourceIdentity(
        revision_id="rev",
        media_id="media",
        object_key="source.wav",
        object_version="v1",
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        sample_rate=rate,
        channels=samples.shape[1],
        frames=len(samples) if expected_frames is None else expected_frames,
    )
    guard = ResourceGuard(
        ResourceBudget(10_000_000, 10_000_000, 200000),
        tmp_path,
        time.monotonic() + 10,
        threading.Event(),
    )
    return path, source, guard


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 24000, 32000, 44100, 48000, 96000])
def test_streamed_scan_preserves_short_tail_and_stereo_evidence(tmp_path, rate):
    samples = np.sin(np.arange(70001) * 0.1) * 0.1
    samples = np.column_stack((samples, -samples))
    result = SourceInspector.inspect(*inspect_fixture(tmp_path, samples, rate))
    assert result.frames == 70001
    assert result.channels == 2
    assert result.sample_rate == rate
    assert result.channel_correlation == pytest.approx(-1)
    assert result.peaks[0] == pytest.approx(0.1)


def test_silence_correlation_is_unavailable(tmp_path):
    result = SourceInspector.inspect(*inspect_fixture(tmp_path, np.zeros((7, 2))))
    assert result.channel_correlation is None
    assert result.rms == (0, 0)


def test_source_mutation_is_rejected(tmp_path):
    path, source, guard = inspect_fixture(tmp_path, np.zeros((10, 1)))
    with path.open("ab") as output:
        output.write(b"changed")
    with pytest.raises(CleanExecutionError) as error:
        SourceInspector.inspect(path, source, guard)
    assert error.value.code == ErrorCode.SOURCE_MISMATCH


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_invalid_samples_rejected(tmp_path, value):
    with pytest.raises(CleanExecutionError) as error:
        SourceInspector.inspect(*inspect_fixture(tmp_path, np.array([[value], [0.0]])))
    assert error.value.code == ErrorCode.INVALID_AUDIO


def test_declared_frame_mismatch_rejected(tmp_path):
    with pytest.raises(CleanExecutionError):
        SourceInspector.inspect(*inspect_fixture(tmp_path, np.zeros((20, 1)), expected_frames=30))


def test_native_scan_is_not_run_in_model_worker(tmp_path, monkeypatch):
    def forbidden(*args):
        pytest.fail("native decoder ran in parent process")

    monkeypatch.setattr(SourceInspector, "_inspect_local", forbidden)
    result = SourceInspector.inspect(*inspect_fixture(tmp_path, np.zeros((20, 1))))
    assert result.frames == 20


def test_inspection_does_not_inherit_service_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "private-test-token")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "private-storage-key")
    monkeypatch.setenv("LD_PRELOAD", "/private/not-a-library")
    original = CancellableProcessRunner.run

    def checked(self, argv, guard, *, env=None):
        assert env is not None
        assert not {"HF_TOKEN", "AWS_SECRET_ACCESS_KEY", "LD_PRELOAD"} & env.keys()
        return original(self, argv, guard, env=env)

    monkeypatch.setattr(CancellableProcessRunner, "run", checked)
    SourceInspector.inspect(*inspect_fixture(tmp_path, np.zeros((20, 1))))


@pytest.mark.parametrize("cancel", [False, True])
def test_supervision_interrupts_stuck_native_scan(tmp_path, monkeypatch, cancel):
    path, source, guard = inspect_fixture(tmp_path, np.zeros((20, 1)))
    original = CancellableProcessRunner.run
    pid_file = tmp_path / "decoder.pid"

    def blocked(self, argv, worker_guard, *, env=None):
        assert argv[2] == "hear.runtime.cleaner.inspection_worker"
        return original(
            self,
            [
                sys.executable,
                "-c",
                "import os,time; from pathlib import Path; "
                "Path('decoder.pid').write_text(str(os.getpid())); time.sleep(60)",
            ],
            worker_guard,
            env=env,
        )

    monkeypatch.setattr(CancellableProcessRunner, "run", blocked)
    timer = None
    if cancel:
        timer = threading.Timer(0.5, guard.cancelled.set)
        timer.start()
    else:
        guard.deadline = time.monotonic() + 0.5
    started = time.monotonic()
    try:
        with pytest.raises(CleanExecutionError) as error:
            SourceInspector.inspect(path, source, guard)
        assert error.value.code == (ErrorCode.CANCELLED if cancel else ErrorCode.DEADLINE_EXCEEDED)
        assert time.monotonic() - started < 3
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid_file.read_text()), 0)
    finally:
        if timer:
            timer.cancel()
            timer.join()


@pytest.mark.parametrize("response", [b"private malformed data", b'{"frames":2}', b"null"])
def test_malformed_child_response_is_sanitized(tmp_path, monkeypatch, response):
    monkeypatch.setattr(CancellableProcessRunner, "run", lambda *a, **k: response)
    with pytest.raises(CleanExecutionError) as error:
        SourceInspector.inspect(*inspect_fixture(tmp_path, np.zeros((20, 1))))
    assert error.value.code == ErrorCode.INVALID_AUDIO
    assert "private" not in str(error.value)
