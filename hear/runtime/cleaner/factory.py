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
from hear.runtime.cleaner.model_registry import CertifiedRuntime, EngineRegistry
from hear.runtime.cleaner.subprocesses import CancellableProcessRunner
from hear.runtime.cleaner.worker_lease import WorkerLease
from hear.services.magic_clean.artifacts import ArtifactWriter, ImmutableArtifactStore
from hear.services.magic_clean.contracts import CleanExecutionError, RuntimeIdentity
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

    def close(self) -> None:
        # Refuses closure while an attempt is active. Session cleanup has finished
        # before executor releases the attempt slot. No model cache is owned here.
        self.executor.worker_lease.close()

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
        speech: SpeechRiskAnalyser | None = None,
    ) -> CleanerWorker:
        """Assemble without downloading/loading models, probing or contacting storage.

        The lock directory must already exist on a shared local mount. Never invent
        certifications here: unavailable/uncertified profiles remain absent. The CPU
        noise-profile lane is separate from the serialized DF3/SAM GPU lane.
        """
        if lane not in ("cpu", "gpu"):
            raise ValueError("invalid cleaner lane")
        for runtime in runtimes:
            expected_lane = "cpu" if runtime.identity.engine == "noise_profile" else "gpu"
            if expected_lane != lane:
                raise ValueError("certified runtime belongs to a different worker lane")
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
            )
            return CleanerWorker(executor)
        except BaseException:
            lease.close()
            raise
