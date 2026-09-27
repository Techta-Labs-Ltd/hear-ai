import importlib
import threading
from pathlib import Path
from typing import Protocol
from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.runtime.cleaner.sam_official import SamOfficialPipeline
from hear.services.magic_clean.contracts import (
    CleanExecutionError,
    CleanPlan,
    ErrorCode,
    RuntimeIdentity,
)


class SamBackend(Protocol):
    pipeline: SamOfficialPipeline
    def close(self) -> None: ...


class SamBackendFactory(Protocol):
    def validate_identity(self, identity: RuntimeIdentity) -> None: ...
    def open(self, guard: ResourceGuard) -> SamBackend: ...
    def close(self) -> None: ...


class SamEngine:
    @staticmethod
    def native_failure(exc):
        if isinstance(exc, CleanExecutionError):
            return CleanExecutionError(
                exc.code, str(exc), worker_restart_required=exc.worker_restart_required
            )
        torch = importlib.import_module("torch")
        exhausted = isinstance(exc, (MemoryError, torch.cuda.OutOfMemoryError))
        if exhausted or isinstance(exc, RuntimeError):
            return CleanExecutionError(
                ErrorCode.RESOURCE_EXHAUSTED if exhausted else ErrorCode.ENGINE_UNAVAILABLE,
                "SAM native runtime failed; worker requires process restart",
                worker_restart_required=True,
            )
        return None

    def __init__(
        self,
        identity: RuntimeIdentity,
        factory: SamBackendFactory,
    ):
        if identity.engine != "sam_audio_base":
            raise ValueError("SAM engine identity required")
        factory.validate_identity(identity)
        self._identity = identity
        self.factory = factory
        self._lease = threading.Lock()
        self._unhealthy = threading.Event()
        self._closed = False

    @property
    def identity(self) -> RuntimeIdentity:
        return self._identity

    def open_session(self, plan: CleanPlan, guard: ResourceGuard):
        guard.check()
        if self._closed:
            raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "SAM engine is closed")
        if self._unhealthy.is_set():
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE,
                "SAM worker requires process restart",
                worker_restart_required=True,
            )
        if plan.runtime != self.identity or plan.profile != "sam_audio":
            raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "SAM runtime unavailable")
        if not self._lease.acquire(blocking=False):
            raise CleanExecutionError(ErrorCode.RESOURCE_EXHAUSTED, "SAM session occupied")
        backend = None
        try:
            self.factory.validate_identity(self.identity)
            backend = self.factory.open(guard)
            guard.check()
            prompt_supported = plan.prompt_text is None or bool(plan.prompt_text.strip())
            if not prompt_supported or plan.channel_policy not in ("mono", "validated_dual_mono"):
                raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "unsupported SAM plan")
            return SamSession(
                plan,
                guard,
                backend,
                self._lease,
                self._unhealthy,
                self.factory.validate_identity,
            )
        except BaseException as exc:
            failure = self.native_failure(exc) if isinstance(exc, Exception) else None
            if failure is not None and failure.worker_restart_required:
                self._unhealthy.set()
            try:
                if backend is not None:
                    backend.close()
            except Exception:
                self._unhealthy.set()
                if failure is not None:
                    failure = CleanExecutionError(
                        failure.code,
                        "SAM session construction cleanup failed; worker requires process restart",
                        worker_restart_required=True,
                    )
                else:
                    failure = CleanExecutionError(
                        ErrorCode.ENGINE_UNAVAILABLE,
                        "SAM session construction cleanup failed; worker requires process restart",
                        worker_restart_required=True,
                    )
            finally:
                self._lease.release()
            if failure is None:
                raise

        raise failure

    def close(self) -> None:
        if self._closed:
            return
        if not self._lease.acquire(blocking=False):
            raise CleanExecutionError(
                ErrorCode.RESOURCE_EXHAUSTED,
                "SAM engine still has an active session",
                worker_restart_required=True,
            )
        try:
            failed = False
            try:
                self.factory.close()
            except BaseException:
                failed = True
            self._closed = True
            if failed:
                raise CleanExecutionError(
                    ErrorCode.ENGINE_UNAVAILABLE,
                    "SAM engine cleanup failed; worker requires process restart",
                    worker_restart_required=True,
                )
        finally:
            self._lease.release()


class SamSession:
    def __init__(
        self,
        plan,
        guard,
        backend,
        lease,
        unhealthy,
        validate_runtime,
    ):
        self.plan, self.guard, self.backend = plan, guard, backend
        self._lease, self._unhealthy = lease, unhealthy
        self._active = threading.Lock()
        self._closed = self._used = False
        self._validate_runtime = validate_runtime

    def process(
        self, source: Path, destination: Path, plan: CleanPlan, guard: ResourceGuard
    ) -> None:
        if not self._active.acquire(blocking=False):
            raise CleanExecutionError(ErrorCode.RESOURCE_EXHAUSTED, "SAM session is processing")
        failure = None
        try:
            guard.check()
            if self._closed or self._used or plan != self.plan or guard is not self.guard:
                raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "invalid SAM session")
            self._validate_runtime(self.plan.runtime)
            self._used = True
            self.backend.pipeline.separate_plan(
                source,
                destination,
                plan=plan,
                expected_runtime=self.plan.runtime,
                guard=guard,
            )
        except Exception as exc:
            failure = SamEngine.native_failure(exc)
            if failure is None:
                raise
            if failure.worker_restart_required:
                self._unhealthy.set()
        finally:
            self._active.release()
        if failure is not None:
            raise failure

    def close(self) -> None:
        if not self._active.acquire(blocking=False):
            raise CleanExecutionError(ErrorCode.RESOURCE_EXHAUSTED, "SAM session is processing")
        try:
            if not self._closed:
                self._closed = True
                try:
                    self.backend.close()
                except BaseException:
                    self._unhealthy.set()
                    raise
                finally:
                    self._lease.release()
        finally:
            self._active.release()
