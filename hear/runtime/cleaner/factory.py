"""Explicit cleaner-only assembly; no environment loading or transport startup.

Call from a dedicated worker process after deployment configuration is validated.
Certificates, offline engine loaders and scoped storage are trusted inputs, not
ticket fields. Models belong to attempt sessions and must close with the session;
loaders that cache models outside sessions require a separate teardown design.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Literal

from hear.runtime.cleaner.executor import CleanExecutor
from hear.runtime.cleaner.gpu_admission import GpuAdmissionController
from hear.runtime.cleaner.model_registry import CertifiedRuntime, EngineRegistry
from hear.runtime.cleaner.subprocesses import CancellableProcessRunner
from hear.runtime.cleaner.worker_lease import WorkerLease
from hear.services.magic_clean.artifacts import ArtifactWriter, ImmutableArtifactStore
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode, RuntimeIdentity
from hear.services.magic_clean.engines.base import CleanEngine
from hear.services.magic_clean.inspection import SourceInspector
from hear.services.magic_clean.mastering import AudioMasteringService
from hear.services.magic_clean.quality import AudioQualityGate, SpeechRiskAnalyser


class CleanerWorker:
    """Owns a lane until all active attempt work has stopped.

    Capabilities are a health snapshot, not admission, queue state or a resource
    reservation. Ingress must authenticate them and the executor's authorizer.
    """

    def __init__(self, executor: CleanExecutor):
        self.executor = executor

    def capabilities(self) -> dict:
        snapshot = self.executor.registry.capabilities()
        try:
            lease = self.executor.worker_lease
            lease.assert_owned(lease.lane)
        except CleanExecutionError:
            for profile in snapshot["profiles"]:
                if profile["runtime"] is not None:
                    profile["ready"] = False
                    profile["reason"] = "worker_unavailable"
            return snapshot
        admission = self.executor.gpu_admission
        for profile in snapshot["profiles"]:
            runtime_info = profile["runtime"]
            if not profile["ready"] or runtime_info is None:
                continue
            runtime = self.executor.registry.runtime_for_engine(runtime_info["engine"])
            if runtime is None or runtime.lane == "cpu":
                continue
            if admission is None:
                profile["ready"] = False
                profile["reason"] = ErrorCode.ENGINE_UNAVAILABLE.value
                continue
            try:
                admission.admit(runtime)
            except CleanExecutionError as error:
                profile["ready"] = False
                profile["reason"] = error.code.value
        return snapshot

    def close(self) -> None:
        self.executor.worker_lease.retire(self.executor.registry.close)

    def __enter__(self):
        lease = self.executor.worker_lease
        lease.assert_owned(lease.lane)
        return self

    def __exit__(self, *args):
        self.close()


class CleanerWorkerFactory:
    @staticmethod
    def build(
        *,
        lock_directory: Path,
        lane: Literal["gpu", "cpu"],
        runtimes: tuple[CertifiedRuntime, ...],
        loaders: dict[str, Callable[[], CleanEngine]],
        readiness: dict[str, Callable[[RuntimeIdentity], bool]],
        store: ImmutableArtifactStore,
        gpu_admission: GpuAdmissionController | None = None,
        speech: SpeechRiskAnalyser | None = None,
    ) -> CleanerWorker:
        """Assemble without downloading/loading models, probing or contacting storage.

        The lock directory must already exist on a shared local mount. Never invent
        certifications here: unavailable/uncertified profiles remain absent.
        """
        if lane not in ("cpu", "gpu"):
            raise ValueError("invalid cleaner lane")
        for runtime in runtimes:
            if runtime.lane != lane:
                raise ValueError("certified runtime belongs to a different worker lane")
        if lane == "gpu" and gpu_admission is None:
            raise ValueError("GPU cleaner worker requires device-memory admission")
        if lane == "cpu" and gpu_admission is not None:
            raise ValueError("CPU cleaner worker must not configure GPU admission")
        registry = EngineRegistry(runtimes, loaders, readiness)
        lease = WorkerLease(lock_directory, lane)
        try:
            executor = CleanExecutor(
                SourceInspector(),
                registry,
                AudioQualityGate(speech),
                AudioMasteringService(CancellableProcessRunner()),
                ArtifactWriter(store),
                lease,
                gpu_admission,
            )
            return CleanerWorker(executor)
        except BaseException:
            lease.close()
            raise
