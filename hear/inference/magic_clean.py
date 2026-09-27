from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from hear.runtime.cleaner.asset_probe import PinnedAssetProbe, PinnedAssetSet
from hear.runtime.cleaner.factory import CleanerWorker
from hear.runtime.cleaner.model_registry import CertifiedRuntime
from hear.runtime.roles import WorkerRole
from hear.services.magic_clean.artifacts import StoredObject
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RuntimeLimits(StrictModel):
    evidence_path: str = Field(min_length=1, max_length=4096)
    evidence_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    max_frames: int = Field(gt=0)
    max_input_bytes: int = Field(gt=0)
    sample_rates: tuple[int, ...]
    channels: tuple[Literal[1, 2], ...]
    certified_peak_device_bytes: int = Field(ge=0)


class NaturalCertification(StrictModel):
    limits: RuntimeLimits
    config_path: str
    config_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    checkpoint_path: str
    checkpoint_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    package_versions: tuple[tuple[str, str], ...]
    device: Literal["cpu", "cuda:0"]
    block_frames: int = Field(ge=48000, le=480000)
    context_frames: int = Field(ge=4800, le=240000)


class CleanerCertifications(StrictModel):
    natural: NaturalCertification | None = None


class DisabledArtifactStore:
    def create(self, *args, **kwargs) -> StoredObject:
        raise CleanExecutionError(ErrorCode.STORAGE_FAILED, "attempt artifact store required")


class MagicCleanRuntimeFactory:
    def __init__(
        self,
        certification_path: Path,
        lock_directory: Path,
        expected_certification_sha256: str,
    ) -> None:
        self._certification_path = certification_path
        self._lock_directory = lock_directory
        if (
            not isinstance(expected_certification_sha256, str)
            or len(expected_certification_sha256) != 64
            or any(value not in "0123456789abcdef" for value in expected_certification_sha256)
        ):
            raise ValueError("cleaner certification requires a pinned SHA-256")
        payload = PinnedAssetProbe.read_regular(
            certification_path,
            maximum_bytes=64 * 1024,
        )
        if hashlib.sha256(payload).hexdigest() != expected_certification_sha256:
            raise RuntimeError("magic_clean_certification_digest_mismatch")
        self._certifications = CleanerCertifications.model_validate_json(payload)
        self._owned: list[object] = []

    def build(self, role: WorkerRole) -> CleanerWorker:
        self._lock_directory.mkdir(parents=True, exist_ok=True)
        if role == WorkerRole.MAGIC_CLEAN_NATURAL:
            return self._build_natural()
        raise RuntimeError("unsupported_magic_clean_role")

    def close(self) -> None:
        for item in reversed(self._owned):
            close = getattr(item, "close", None)
            if callable(close):
                close()
        self._owned.clear()

    def _build_natural(self) -> CleanerWorker:
        from hear.runtime.cleaner.deepfilter_loader import (
            PinnedDeepFilterAssets,
            PinnedDeepFilterFactory,
        )
        from hear.runtime.cleaner.factory import CleanerWorkerFactory
        from hear.runtime.cleaner.gpu_admission import GpuAdmissionController, NvidiaSmiMemoryProbe
        from hear.services.magic_clean.engines.deepfilter import ContextualPolicy, DeepFilterEngine

        cert = self._certifications.natural
        if cert is None:
            raise RuntimeError("magic_clean_natural_not_certified")
        assets = PinnedDeepFilterAssets(
            Path(cert.config_path),
            cert.config_sha256,
            Path(cert.checkpoint_path),
            cert.checkpoint_sha256,
            cert.package_versions,
            cert.device,
        )
        pinned_assets = PinnedAssetSet(
            (
                (assets.config_path, assets.config_sha256, 1024 * 1024),
                (assets.checkpoint_path, assets.checkpoint_sha256, None),
            )
        )
        policy = ContextualPolicy(cert.block_frames, cert.context_frames)
        factory = PinnedDeepFilterFactory(assets)
        identity = factory.identity(policy.digest)
        runtime = self._runtime(cert.limits, identity, "gpu" if cert.device == "cuda:0" else "cpu")

        def loader():
            return DeepFilterEngine(identity, factory, policy)

        def ready(candidate):
            if candidate != identity:
                return False
            pinned_assets.verify()
            PinnedAssetProbe.packages(cert.package_versions)
            factory.validate_identity(identity)
            return True

        return CleanerWorkerFactory.build(
            lock_directory=self._lock_directory,
            lane="gpu" if cert.device == "cuda:0" else "cpu",
            runtimes=(runtime,),
            loaders={"deepfilternet3": loader},
            readiness={"deepfilternet3": ready},
            store=DisabledArtifactStore(),
            gpu_admission=(
                GpuAdmissionController(NvidiaSmiMemoryProbe()) if cert.device == "cuda:0" else None
            ),
        )

    @staticmethod
    def _runtime(
        limits: RuntimeLimits,
        identity,
        lane: Literal["cpu", "gpu"],
    ) -> CertifiedRuntime:
        PinnedAssetProbe.sha256(
            Path(limits.evidence_path),
            limits.evidence_sha256,
            maximum_bytes=16 * 1024 * 1024,
        )
        return CertifiedRuntime(
            identity,
            limits.evidence_sha256,
            limits.max_frames,
            limits.max_input_bytes,
            limits.sample_rates,
            limits.channels,
            lane,
            limits.certified_peak_device_bytes,
        )

    @staticmethod
    def _verify_file(path: Path, expected: str) -> None:
        PinnedAssetProbe.sha256(path, expected)
