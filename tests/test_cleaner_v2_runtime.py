import errno
import json
import os
import sys
import threading
import time
from datetime import UTC, datetime, timedelta

import pytest

from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.runtime.cleaner.subprocesses import CancellableProcessRunner
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


def guard_for(tmp_path, seconds=5, scratch=4096):
    return ResourceGuard(
        ResourceBudget(scratch, 4096, 48000),
        tmp_path,
        time.monotonic() + seconds,
        threading.Event(),
    )


def test_child_diagnostics_are_bounded_without_pipe_deadlock(tmp_path):
    result = CancellableProcessRunner(64).run(
        [sys.executable, "-c", "import sys; sys.stderr.write('x' * 200000)"], guard_for(tmp_path)
    )
    assert result == b"x" * 64


def test_child_environment_does_not_inherit_server_secrets_or_controls(tmp_path, monkeypatch):
    inherited = (
        "HF_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
        "HEAR_SERVICE_KEY",
        "HTTPS_PROXY",
        "PYTHONPATH",
        "FFREPORT",
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
    )
    for key in inherited:
        monkeypatch.setenv(key, "private-server-value")
    monkeypatch.setenv("OMP_NUM_THREADS", "99")
    result = CancellableProcessRunner().run(
        [
            sys.executable,
            "-I",
            "-c",
            "import json,os,sys; sys.stderr.write(json.dumps(dict(os.environ)))",
        ],
        guard_for(tmp_path),
    )
    environment = json.loads(result)
    assert all(key not in environment for key in inherited)
    assert environment["PATH"] == os.defpath
    assert environment["LANG"] == "C.UTF-8"
    assert environment["OMP_NUM_THREADS"] == "1"
    assert environment["OPENBLAS_NUM_THREADS"] == "1"


def test_explicit_child_environment_is_not_merged_with_parent(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "private-server-value")
    supplied = {"CLEANER_TEST_SETTING": "explicit"}
    result = CancellableProcessRunner().run(
        [
            sys.executable,
            "-I",
            "-c",
            "import json,os,sys; sys.stderr.write(json.dumps(dict(os.environ)))",
        ],
        guard_for(tmp_path),
        env=supplied,
    )
    environment = json.loads(result)
    assert environment["CLEANER_TEST_SETTING"] == "explicit"
    assert "HF_TOKEN" not in environment
    assert "PATH" not in environment
    assert supplied == {"CLEANER_TEST_SETTING": "explicit"}


def test_ffmpeg_does_not_create_inherited_report_file(tmp_path, monkeypatch):
    report = tmp_path / "private-codec-report.log"
    monkeypatch.setenv("FFREPORT", f"file={report}:level=48")
    CancellableProcessRunner().run(
        ["ffmpeg", "-hide_banner", "-version"], guard_for(tmp_path, scratch=1_000_000)
    )
    assert not report.exists()


def test_deadline_stops_process_group_even_after_leader_exits(tmp_path):
    started = time.monotonic()
    with pytest.raises(CleanExecutionError) as error:
        CancellableProcessRunner().run(
            [
                sys.executable,
                "-c",
                "import subprocess,sys; "
                "subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])",
            ],
            guard_for(tmp_path, seconds=0.2),
        )
    assert error.value.code == ErrorCode.DEADLINE_EXCEEDED
    assert time.monotonic() - started < 3


def test_cancellation_prevents_child_launch(tmp_path):
    guard = guard_for(tmp_path)
    guard.cancelled.set()
    with pytest.raises(CleanExecutionError) as error:
        CancellableProcessRunner().run(["nonexistent-executable"], guard)
    assert error.value.code == ErrorCode.CANCELLED


def test_disk_growth_terminates_child(tmp_path):
    with pytest.raises(CleanExecutionError) as error:
        CancellableProcessRunner().run(
            [
                sys.executable,
                "-c",
                "import time; open('pcm','wb').write(b'x'*8192); time.sleep(60)",
            ],
            guard_for(tmp_path),
        )
    assert error.value.code == ErrorCode.RESOURCE_EXHAUSTED


def test_reservation_uses_all_channels_and_copies(tmp_path):
    guard = guard_for(tmp_path)
    assert guard.preflight_pcm(100, 2, 3, 100) == 2500
    with pytest.raises(CleanExecutionError):
        guard.preflight_pcm(1000, 2, 3, 100)


def test_gpu_cap_cannot_be_raised():
    with pytest.raises(ValueError):
        ResourceBudget(1, 1, 1, gpu_limit_bytes=12 * 1024**3)


def test_child_failure_does_not_expose_diagnostics(tmp_path):
    with pytest.raises(CleanExecutionError) as error:
        CancellableProcessRunner().run(
            [sys.executable, "-c", "import sys; sys.stderr.write('secret'); sys.exit(1)"],
            guard_for(tmp_path),
        )
    assert "secret" not in str(error.value)
    assert error.value.code == ErrorCode.PROCESS_FAILED


def test_ticket_deadline_only_tightens_monotonic_timeout(tmp_path):
    guard = guard_for(tmp_path, seconds=60)
    deadline = datetime.now(UTC) + timedelta(seconds=2)
    guard.bind_deadline(deadline)
    bound = guard.deadline
    assert bound < time.monotonic() + 3
    guard.bind_deadline(deadline + timedelta(hours=1))
    assert guard.deadline == bound
    assert guard.wall_deadline == deadline


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_worker_deadline_rejected(tmp_path, deadline):
    with pytest.raises(ValueError):
        ResourceGuard(ResourceBudget(1000, 1000, 1000), tmp_path, deadline, threading.Event())


def test_forward_wall_clock_jump_expires_bound_guard(tmp_path, monkeypatch):
    guard = guard_for(tmp_path, seconds=60)
    deadline = datetime.now(UTC) + timedelta(seconds=30)
    guard.bind_deadline(deadline)

    class ForwardClock:
        @staticmethod
        def now(zone):
            return deadline + timedelta(seconds=1)

    monkeypatch.setattr("hear.runtime.cleaner.resource_guard.datetime", ForwardClock)
    with pytest.raises(CleanExecutionError) as error:
        guard.check()
    assert error.value.code == ErrorCode.DEADLINE_EXCEEDED


def test_naive_ticket_deadline_rejected(tmp_path):
    with pytest.raises(ValueError):
        guard_for(tmp_path).bind_deadline(datetime(2030, 1, 1))


def test_backward_wall_clock_jump_cannot_extend_bound_budget(tmp_path, monkeypatch):
    guard = guard_for(tmp_path, seconds=60)
    deadline = datetime.now(UTC) + timedelta(seconds=30)
    guard.bind_deadline(deadline)
    bound = guard.deadline

    class BackwardClock:
        @staticmethod
        def now(zone):
            return deadline - timedelta(hours=2)

    monkeypatch.setattr("hear.runtime.cleaner.resource_guard.datetime", BackwardClock)
    guard.bind_deadline(deadline)
    assert guard.deadline == bound


@pytest.mark.parametrize(
    "number,expected",
    [
        (errno.ENOENT, ErrorCode.ENGINE_UNAVAILABLE),
        (errno.EACCES, ErrorCode.ENGINE_UNAVAILABLE),
        (errno.ENOEXEC, ErrorCode.ENGINE_UNAVAILABLE),
        (errno.ENOMEM, ErrorCode.RESOURCE_EXHAUSTED),
        (errno.EAGAIN, ErrorCode.RESOURCE_EXHAUSTED),
        (errno.EMFILE, ErrorCode.RESOURCE_EXHAUSTED),
        (errno.ENFILE, ErrorCode.RESOURCE_EXHAUSTED),
        (errno.EIO, ErrorCode.PROCESS_FAILED),
    ],
)
def test_child_launch_failures_are_typed_and_sanitized(tmp_path, monkeypatch, number, expected):
    def fail(*args, **kwargs):
        raise OSError(number, "private-source-path-and-grant")

    monkeypatch.setattr("hear.runtime.cleaner.subprocesses.subprocess.Popen", fail)
    with pytest.raises(CleanExecutionError) as error:
        CancellableProcessRunner().run(["ffmpeg"], guard_for(tmp_path))
    assert error.value.code == expected
    assert "private-source" not in str(error.value)
    assert error.value.__suppress_context__


def test_real_missing_executable_is_typed_unavailable(tmp_path):
    with pytest.raises(CleanExecutionError) as error:
        CancellableProcessRunner().run([str(tmp_path / "missing-codec")], guard_for(tmp_path))
    assert error.value.code == ErrorCode.ENGINE_UNAVAILABLE
