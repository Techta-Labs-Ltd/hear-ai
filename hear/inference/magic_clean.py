from __future__ import annotations

import hashlib
import tempfile
import threading
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from hear.runtime.cleaner.deepfilter_loader import (
    PinnedDeepFilterAssets,
    PinnedDeepFilterFactory,
)
from hear.runtime.cleaner.factory import CleanerWorker, CleanerWorkerFactory
from hear.runtime.cleaner.model_registry import CertifiedRuntime
from hear.runtime.cleaner.noise_reference import SpeechAwareNoiseReferenceAnalyser
from hear.runtime.cleaner.resampling import AudioResampler
from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.runtime.cleaner.sam_loader import PinnedSamAssets, PinnedSamFactory
from hear.runtime.cleaner.sam_prompt_cache import SamPromptCache, SamPromptIdentity
from hear.runtime.cleaner.speech_activity import CpuSpeechActivity, SpeechActivityPolicy
from hear.runtime.cleaner.speech_risk import SpeechRiskComparison
from hear.runtime.cleaner.subprocesses import CancellableProcessRunner
from hear.runtime.roles import WorkerRole
from hear.services.magic_clean.artifacts import StoredObject
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode
from hear.services.magic_clean.engines.deepfilter import ContextualPolicy, DeepFilterEngine
from hear.services.magic_clean.engines.noise_profile import NoiseProfileEngine
from hear.services.magic_clean.engines.sam_audio import SamEngine


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RuntimeLimits(StrictModel):
    evidence_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    max_frames: int = Field(gt=0)
    max_input_bytes: int = Field(gt=0)
    sample_rates: tuple[int, ...]
    channels: tuple[Literal[1, 2], ...]


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


class SpeechCertification(StrictModel):
    model_path: str
    model_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    onnxruntime_version: str
    numpy_version: str
    threshold: float = Field(gt=0, lt=1)


class MusicCertification(StrictModel):
    limits: RuntimeLimits
    speech: SpeechCertification


class PromptCertification(StrictModel):
    path: str
    prompt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    precision_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    embedding_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    mask_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    tokens: int = Field(ge=1, le=512)


class VoiceFocusCertification(StrictModel):
    limits: RuntimeLimits
    sam_source: str
    codec_source: str
    config_path: str
    checkpoint_path: str
    optional_manifest_path: str
    prompt: PromptCertification
    device: Literal["cpu", "cuda:0"]


class CleanerCertifications(StrictModel):
    natural: NaturalCertification | None = None
    voice_focus: VoiceFocusCertification | None = None
    music_atmosphere: MusicCertification | None = None


class DisabledArtifactStore:
    def create(self, *args, **kwargs) -> StoredObject:
        raise CleanExecutionError(ErrorCode.STORAGE_FAILED, "attempt artifact store required")


class MagicCleanRuntimeFactory:
    def __init__(
        self,
        certification_path: Path,
        lock_directory: Path,
    ) -> None:
        self._certification_path = certification_path
        self._lock_directory = lock_directory
        self._certifications = CleanerCertifications.model_validate_json(
            certification_path.read_text()
        )
        self._owned: list[object] = []

    def build(self, role: WorkerRole) -> CleanerWorker:
        self._lock_directory.mkdir(parents=True, exist_ok=True)
        if role == WorkerRole.MAGIC_CLEAN_NATURAL:
            return self._build_natural()
        if role == WorkerRole.MAGIC_CLEAN_VOICE_FOCUS:
            return self._build_voice_focus()
        if role == WorkerRole.MAGIC_CLEAN_MUSIC_ATMOSPHERE:
            return self._build_music()
        raise RuntimeError("unsupported_magic_clean_role")

    def close(self) -> None:
        for item in reversed(self._owned):
            close = getattr(item, "close", None)
            if callable(close):
                close()
        self._owned.clear()

    def _build_natural(self) -> CleanerWorker:
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
        policy = ContextualPolicy(cert.block_frames, cert.context_frames)
        factory = PinnedDeepFilterFactory(assets)
        identity = factory.identity(policy.digest)
        runtime = self._runtime(cert.limits, identity)

        def loader():
            return DeepFilterEngine(identity, factory, policy)

        def ready(candidate):
            if candidate != identity:
                return False
            self._verify_file(Path(cert.config_path), cert.config_sha256)
            self._verify_file(Path(cert.checkpoint_path), cert.checkpoint_sha256)
            factory.validate_identity(identity)
            return True

        return CleanerWorkerFactory.build(
            lock_directory=self._lock_directory,
            lane="gpu" if cert.device == "cuda:0" else "cpu",
            runtimes=(runtime,),
            loaders={"deepfilternet3": loader},
            readiness={"deepfilternet3": ready},
            store=DisabledArtifactStore(),
        )

    def _build_music(self) -> CleanerWorker:
        cert = self._certifications.music_atmosphere
        if cert is None:
            raise RuntimeError("magic_clean_music_atmosphere_not_certified")
        speech = self._speech(cert.speech)
        analyser = SpeechAwareNoiseReferenceAnalyser(speech)
        identity = NoiseProfileEngine.describe(analyser)
        runtime = self._runtime(cert.limits, identity)

        def loader():
            return NoiseProfileEngine(identity, analyser)

        def ready(candidate):
            return candidate == identity

        risk = SpeechRiskComparison(speech)
        self._owned.append(speech)
        return CleanerWorkerFactory.build(
            lock_directory=self._lock_directory,
            lane="cpu",
            runtimes=(runtime,),
            loaders={"noise_profile": loader},
            readiness={"noise_profile": ready},
            store=DisabledArtifactStore(),
            speech=risk,
        )

    def _build_voice_focus(self) -> CleanerWorker:
        cert = self._certifications.voice_focus
        if cert is None:
            raise RuntimeError("magic_clean_voice_focus_not_certified")
        prompt = SamPromptIdentity(
            prompt_sha256=cert.prompt.prompt_sha256,
            model_sha256=cert.prompt.model_sha256,
            precision_sha256=cert.prompt.precision_sha256,
            embedding_sha256=cert.prompt.embedding_sha256,
            mask_sha256=cert.prompt.mask_sha256,
            tokens=cert.prompt.tokens,
        )
        cache = SamPromptCache.from_files(((prompt, Path(cert.prompt.path)),))
        assets = PinnedSamAssets(
            Path(cert.sam_source),
            Path(cert.codec_source),
            Path(cert.config_path),
            Path(cert.checkpoint_path),
            Path(cert.optional_manifest_path),
        )
        factory = PinnedSamFactory(
            assets,
            prompt,
            cache,
            device=cert.device,
        )
        identity = factory.identity
        runtime = self._runtime(cert.limits, identity)

        def loader():
            return SamEngine(identity, factory)

        def ready(candidate):
            if candidate != identity:
                return False
            self._verify_file(Path(cert.checkpoint_path), PinnedSamFactory.CHECKPOINT)
            factory.validate_identity(identity)
            return True

        self._owned.append(cache)
        return CleanerWorkerFactory.build(
            lock_directory=self._lock_directory,
            lane="gpu" if cert.device == "cuda:0" else "cpu",
            runtimes=(runtime,),
            loaders={"sam_audio_small": loader},
            readiness={"sam_audio_small": ready},
            store=DisabledArtifactStore(),
        )

    def _speech(self, cert: SpeechCertification) -> CpuSpeechActivity:
        model = Path(cert.model_path)
        self._verify_file(model, cert.model_sha256)
        policy = SpeechActivityPolicy(
            cert.model_sha256,
            cert.onnxruntime_version,
            cert.numpy_version,
            cert.threshold,
        )
        with tempfile.TemporaryDirectory(prefix="hear-speech-bootstrap-") as directory:
            guard = ResourceGuard(
                ResourceBudget(
                    64 * 1024 * 1024,
                    32 * 1024 * 1024,
                    16_000 * 60,
                ),
                Path(directory),
                time.monotonic() + 60,
                threading.Event(),
            )
            return CpuSpeechActivity(
                model,
                policy,
                AudioResampler(CancellableProcessRunner()),
                guard,
            )

    @staticmethod
    def _runtime(limits: RuntimeLimits, identity) -> CertifiedRuntime:
        return CertifiedRuntime(
            identity,
            limits.evidence_sha256,
            limits.max_frames,
            limits.max_input_bytes,
            limits.sample_rates,
            limits.channels,
        )

    @staticmethod
    def _verify_file(path: Path, expected: str) -> None:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("certified_asset_missing")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise RuntimeError("certified_asset_mismatch")
