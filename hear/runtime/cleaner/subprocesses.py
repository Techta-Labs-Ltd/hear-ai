import errno
import os
import selectors
import signal
import subprocess

from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


class CancellableProcessRunner:
    def __init__(self, diagnostic_bytes: int = 16384):
        if diagnostic_bytes < 1 or diagnostic_bytes > 1_048_576:
            raise ValueError("invalid diagnostic limit")
        self.diagnostic_bytes = diagnostic_bytes

    def run(
        self, argv: list[str], guard: ResourceGuard, *, env: dict[str, str] | None = None
    ) -> bytes:
        guard.check()
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
                    guard.check()
                    for key, _ in selector.select(timeout=0.05):
                        chunk = os.read(key.fd, 8192)
                        if chunk:
                            tail.extend(chunk)
                            del tail[: -self.diagnostic_bytes]
                        else:
                            selector.unregister(key.fileobj)
                    if process.poll() is not None and not selector.get_map():
                        break
            guard.check()
            if process.returncode:
                # Do not embed paths, signed URLs or raw child diagnostics in errors.
                raise CleanExecutionError(ErrorCode.PROCESS_FAILED, "audio subprocess failed")
            return bytes(tail)
        finally:
            # Stop descendants even if their group leader has already exited.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            if process.stderr:
                process.stderr.close()
