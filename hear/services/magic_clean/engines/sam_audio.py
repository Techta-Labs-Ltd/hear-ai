"""SAM executor/session adapter; admission and pinned loading stay explicit."""

import importlib
import threading
from pathlib import Path
from typing import Protocol

from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.runtime.cleaner.sam_pipeline import SamSeparationPipeline
from hear.runtime.cleaner.sam_prompt_cache import SamPromptCache, SamPromptIdentity
from hear.services.magic_clean.contracts import (
    CleanExecutionError,
    CleanPlan,
    ErrorCode,
    RuntimeIdentity,
)


class SamBackend(Protocol):
    pipeline: SamSeparationPipeline
    prompt: SamPromptIdentity
    cache: SamPromptCache

    def close(self) -> None: ...


class SamBackendFactory(Protocol):
    def validate_identity(self, identity: RuntimeIdentity) -> None:
        """Verify assets, source, precision, solver, RNG, codec and resampling policy."""
        ...

    def open(self, guard: ResourceGuard) -> SamBackend:
        """Return owned attempt state; clean partial loading on failure."""
        ...


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

    def __init__(self, identity: RuntimeIdentity, factory: SamBackendFactory):
        if identity.engine != "sam_audio_small":
            raise ValueError("SAM engine identity required")
        factory.validate_identity(identity)
        self._identity = identity
        self.factory = factory
        self._lease = threading.Lock()
        self._unhealthy = threading.Event()

    @property
    def identity(self) -> RuntimeIdentity:
        return self._identity

    def open_session(self, plan: CleanPlan, guard: ResourceGuard):
        guard.check()
        if self._unhealthy.is_set():
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE,
                "SAM worker requires process restart",
                worker_restart_required=True,
            )
        if plan.runtime != self.identity or plan.profile != "voice_focus":
            raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "SAM runtime unavailable")
        if not self._lease.acquire(blocking=False):
            raise CleanExecutionError(ErrorCode.RESOURCE_EXHAUSTED, "SAM session occupied")
        backend = None
        try:
            self.factory.validate_identity(self.identity)
            backend = self.factory.open(guard)
            guard.check()
            if plan.prompt_sha256 != backend.prompt.prompt_sha256 or plan.channel_policy != "mono":
                raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "unsupported SAM plan")
            return SamSession(
                plan, guard, backend, self._lease, self._unhealthy, self.factory.validate_identity
            )
        except BaseException as exc:
            failure = self.native_failure(exc) if isinstance(exc, Exception) else None
            if failure is not None and failure.worker_restart_required:
                self._unhealthy.set()
            try:
                if backend is not None:
                    backend.close()
            except BaseException:
                self._unhealthy.set()
                raise
            finally:
                self._lease.release()
            if failure is None:
                raise
        # Do not retain native traceback frames containing partially loaded tensors.
        raise failure


class SamSession:
    def __init__(self, plan, guard, backend, lease, unhealthy, validate_runtime):
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
                prompt=self.backend.prompt,
                cache=self.backend.cache,
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
