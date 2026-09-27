import errno
import os
import selectors
import signal
import subprocess
import sys

from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


class CancellableProcessRunner:
    POLICY = "process-group-sigkill-bounded-wait-v2"
    DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 2.0

    def __init__(
        self,
        diagnostic_bytes: int = 16384,
        shutdown_timeout_seconds: float = DEFAULT_SHUTDOWN_TIMEOUT_SECONDS,
    ):
        if diagnostic_bytes < 1 or diagnostic_bytes > 1_048_576:
            raise ValueError("invalid diagnostic limit")
        if not 0 < shutdown_timeout_seconds <= 10:
            raise ValueError("invalid subprocess shutdown timeout")
        self.diagnostic_bytes = diagnostic_bytes
        self.shutdown_timeout_seconds = shutdown_timeout_seconds

    @staticmethod
    def _cleanup_error(primary: BaseException | None) -> CleanExecutionError:
        code = primary.code if isinstance(primary, CleanExecutionError) else ErrorCode.PROCESS_FAILED
        return CleanExecutionError(
            code,
            "audio subprocess cleanup failed; worker requires process restart",
            worker_restart_required=True,
        )

    def _terminate(self, process: subprocess.Popen, primary: BaseException | None) -> None:
        cleanup_failed = False
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            cleanup_failed = True
            try:
                process.kill()
            except OSError:
                pass
        try:
            process.wait(timeout=self.shutdown_timeout_seconds)
        except subprocess.TimeoutExpired:
            cleanup_failed = True
            try:
                process.kill()
            except OSError:
                pass
            try:
                process.wait(timeout=self.shutdown_timeout_seconds)
            except subprocess.TimeoutExpired:
                pass
        if process.poll() is None:
            cleanup_failed = True
        if cleanup_failed:
            raise self._cleanup_error(primary) from primary

    def run(
        self, argv: list[str], guard: ResourceGuard, *, env: dict[str, str] | None = None
    ) -> bytes:
        guard.check_scratch()
        # Codecs consume untrusted media and need no server credentials. Do not
        # inherit token/proxy variables, library injection or FFREPORT settings.
        # An explicit environment is trusted constructor wiring (e.g. the
        # inspection worker's package path), never ticket/user-supplied data.
        child_env = (
            {
                "PATH": os.defpath,
                "LANG": "C.UTF-8",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            }
            if env is None
            else dict(env)
        )
        try:
            process = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,
                cwd=guard.workspace,
                env=child_env,
            )
        except OSError as exc:
            if exc.errno in (errno.ENOMEM, errno.EAGAIN, errno.EMFILE, errno.ENFILE):
                code = ErrorCode.RESOURCE_EXHAUSTED
            elif exc.errno in (errno.ENOENT, errno.EACCES, errno.ENOEXEC):
                code = ErrorCode.ENGINE_UNAVAILABLE
            else:
                code = ErrorCode.PROCESS_FAILED
            # Popen errors can include executable/workspace paths and arguments.
            raise CleanExecutionError(code, "audio subprocess could not start") from None
        tail = bytearray()
        try:
            assert process.stderr is not None
            os.set_blocking(process.stderr.fileno(), False)
            with selectors.DefaultSelector() as selector:
                selector.register(process.stderr, selectors.EVENT_READ)
                while True:
                    guard.check_scratch()
                    for key, _ in selector.select(timeout=0.05):
                        chunk = os.read(key.fd, 8192)
                        if chunk:
                            tail.extend(chunk)
                            del tail[: -self.diagnostic_bytes]
                        else:
                            selector.unregister(key.fileobj)
                    if process.poll() is not None and not selector.get_map():
                        break
            guard.check_scratch()
            if process.returncode:
                # Do not embed paths, signed URLs or raw child diagnostics in errors.
                raise CleanExecutionError(ErrorCode.PROCESS_FAILED, "audio subprocess failed")
            return bytes(tail)
        finally:
            try:
                primary = sys.exception()
                self._terminate(process, primary)
            finally:
                if process.stderr:
                    process.stderr.close()
