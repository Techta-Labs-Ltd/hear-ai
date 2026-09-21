import selectors
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from hear.runtime.cleaner.worker_lease import WorkerLease
from hear.services.magic_clean.contracts import CleanExecutionError


def test_lanes_are_independent_and_lease_survives_multiple_checks(tmp_path):
    with WorkerLease(tmp_path, "gpu") as gpu, WorkerLease(tmp_path, "cpu") as cpu:
        gpu.assert_owned("gpu")
        cpu.assert_owned("cpu")
        with pytest.raises(CleanExecutionError):
            WorkerLease(tmp_path, "gpu")
        gpu.assert_owned("gpu")
        with pytest.raises(CleanExecutionError):
            gpu.assert_owned("cpu")
    assert (tmp_path / "cleaner-gpu.lock").exists()
    with WorkerLease(tmp_path, "gpu") as replacement:
        replacement.assert_owned("gpu")


def test_second_process_is_rejected(tmp_path):
    with WorkerLease(tmp_path, "gpu"):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from pathlib import Path; "
                "from hear.runtime.cleaner.worker_lease import WorkerLease; "
                "import sys; WorkerLease(Path(sys.argv[1]), 'gpu')",
                str(tmp_path),
            ],
            capture_output=True,
            timeout=15,
        )
    assert result.returncode != 0
    assert b"cleaner worker lane unavailable" in result.stderr


def test_process_exit_releases_lock_without_deleting_inode(tmp_path):
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; from hear.runtime.cleaner.worker_lease import WorkerLease; "
            "import sys; lease=WorkerLease(Path(sys.argv[1]), 'gpu'); "
            "print('owned',flush=True); sys.stdin.readline()",
            str(tmp_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            assert selector.select(timeout=15), "child did not acquire worker lane in time"
            assert process.stdout.readline() == b"owned\n"
        inode = (tmp_path / "cleaner-gpu.lock").stat().st_ino
        with pytest.raises(CleanExecutionError):
            WorkerLease(tmp_path, "gpu")
        process.terminate()
        process.wait(timeout=5)
        with WorkerLease(tmp_path, "gpu"):
            assert (tmp_path / "cleaner-gpu.lock").stat().st_ino == inode
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)


def test_symlink_lock_is_rejected_without_touching_target(tmp_path):
    target = tmp_path / "keep"
    target.write_text("private")
    (tmp_path / "cleaner-gpu.lock").symlink_to(target)
    with pytest.raises(CleanExecutionError):
        WorkerLease(tmp_path, "gpu")
    assert target.read_text() == "private"


def test_replaced_inode_and_closed_lease_reject_work(tmp_path):
    with WorkerLease(tmp_path, "gpu") as lease:
        lease.path.rename(tmp_path / "old-lock")
        lease.path.touch()
        with pytest.raises(CleanExecutionError):
            lease.assert_owned("gpu")
    with pytest.raises(CleanExecutionError):
        lease.assert_owned("gpu")


def test_worker_serializes_attempts_and_cannot_close_while_busy(tmp_path):
    with WorkerLease(tmp_path, "gpu") as worker:
        with worker.attempt("gpu"):
            with pytest.raises(CleanExecutionError):
                with worker.attempt("gpu"):
                    pytest.fail("second request was admitted")
            with pytest.raises(CleanExecutionError):
                worker.close()
            worker.assert_owned("gpu")
        with worker.attempt("gpu"):
            worker.assert_owned("gpu")
        # Finishing attempts must not release process/model-cache ownership.
        with pytest.raises(CleanExecutionError):
            WorkerLease(tmp_path, "gpu")


def test_attempt_exception_releases_admission_not_worker_ownership(tmp_path):
    with WorkerLease(tmp_path, "gpu") as worker:
        with pytest.raises(ValueError):
            with worker.attempt("gpu"):
                raise ValueError("injected attempt error")
        with worker.attempt("gpu"):
            worker.assert_owned("gpu")


def test_attempt_admission_is_shared_across_threads(tmp_path):
    entered, release = threading.Event(), threading.Event()
    with WorkerLease(tmp_path, "gpu") as worker, ThreadPoolExecutor(max_workers=1) as pool:

        def first_request():
            with worker.attempt("gpu"):
                entered.set()
                assert release.wait(timeout=5)

        future = pool.submit(first_request)
        try:
            assert entered.wait(timeout=5)
            with pytest.raises(CleanExecutionError):
                with worker.attempt("gpu"):
                    pytest.fail("concurrent thread was admitted")
        finally:
            release.set()
        future.result(timeout=5)
        with worker.attempt("gpu"):
            worker.assert_owned("gpu")
