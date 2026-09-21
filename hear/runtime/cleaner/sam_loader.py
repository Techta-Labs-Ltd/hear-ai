"""Concrete offline SAM factory; device admission is not serving certification."""

import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path

from hear.runtime.cleaner.longform_sam import SolverPolicy
from hear.runtime.cleaner.resampling import AudioResampler
from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.runtime.cleaner.sam_checkpoint import SamCheckpointLoader
from hear.runtime.cleaner.sam_codec_loader import SamCodecBuilder
from hear.runtime.cleaner.sam_core_loader import SamCoreBuilder
from hear.runtime.cleaner.sam_noise import SamNoisePolicy
from hear.runtime.cleaner.sam_pipeline import SamSeparationPipeline
from hear.runtime.cleaner.sam_prompt_cache import SamPromptCache, SamPromptIdentity
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode, RuntimeIdentity


@dataclass(frozen=True)
class PinnedSamAssets:
    sam_source: Path
    codec_source: Path
    config: Path
    checkpoint: Path
    optional_manifest: Path


class LoadedSamBackend:
    def __init__(self, core, codec, pipeline, prompt, cache):
        self._core, self._codec = core, codec
        self.pipeline, self.prompt, self.cache = pipeline, prompt, cache

    def close(self):
        self.pipeline = None
        try:
            if self._codec is not None:
                self._codec.close()
        finally:
            if self._core is not None:
                self._core.close()
        self._core = self._codec = None
        # The immutable prompt cache is borrowed from the factory, not attempt-owned.


class PinnedSamFactory:
    CHECKPOINT = "8c44fda9821fd9f2ec8977304e3c0f55290d9eacb6bbf25b4b8fb1f69c2a8c06"
    OPTIONAL_MANIFEST = "df43a1d8d8306fab6d9e9d70c991eb364efbdacdddac51bdc2e981a914bbed7e"
    PACKAGES = {"torch": "2.8.0+cu128", "numpy": "1.26.4", "einops": "0.8.2", "soundfile": "0.12.1"}

    def __init__(
        self,
        assets: PinnedSamAssets,
        prompt: SamPromptIdentity,
        cache: SamPromptCache,
        *,
        solver: SolverPolicy | None = None,
        codec_tile_frames: int = 4096,
        device: str = "cpu",
    ):
        solver = solver or SolverPolicy(250, 50, 16)
        if solver.steps != 16 or solver.window_frames > 250:
            raise ValueError("unsupported SAM solver policy")
        if type(codec_tile_frames) is not int or not 1 <= codec_tile_frames <= 65536:
            raise ValueError("invalid SAM codec tile policy")
        if device not in ("cpu", "cuda:0"):
            raise ValueError("unsupported SAM device")
        self.device = device
        self.assets, self.prompt, self.cache = assets, prompt, cache
        self.solver, self.codec_tile_frames = solver, codec_tile_frames

    @staticmethod
    def _digest(value) -> str:
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def identity(self) -> RuntimeIdentity:
        precision_policy = {
            "device": self.device,
            "dtype": "float32",
            "autocast": False,
            "mkldnn": True,
            "mkldnn_deterministic": False,
            "deterministic_algorithms": False,
            "float32_matmul_precision": "highest",
        }
        if self.device == "cuda:0":
            precision_policy.update(
                tf32=False, cudnn=True, cudnn_benchmark=False, cudnn_deterministic=False
            )
        precision = self._digest(precision_policy)
        longform = self._digest(
            {
                "version": "sam-file-pipeline-v1",
                "solver": self.solver.digest,
                "codec_tile_frames": self.codec_tile_frames,
                "rng": SamNoisePolicy().digest,
                "resampling": AudioResampler.POLICY,
                "codec_mean_only": True,
                "watermark": "retained-alpha0.25-two-stream-messages",
                "channels": "mono-only",
            }
        )
        runtime = self._digest(
            {
                "loader": (
                    "sam-offline-meta-cpu-v3"
                    if self.device == "cpu"
                    else "sam-offline-meta-cuda-fp32-v1"
                ),
                "fault_policy": "native-oom-or-runtime-fault-requires-worker-restart",
                "checkpoint": self.CHECKPOINT,
                "config": SamCoreBuilder.CONFIG_SHA256,
                "optional_manifest": self.OPTIONAL_MANIFEST,
                "core_sources": SamCoreBuilder.SOURCES,
                "codec_sources": SamCodecBuilder.SOURCES,
                "packages": self.PACKAGES,
                "prompt_cache_identity": self.prompt.digest,
                "precision": precision,
                "longform": longform,
            }
        )
        return RuntimeIdentity(
            engine="sam_audio_small",
            checkpoint_sha256=self.CHECKPOINT,
            runtime_sha256=runtime,
            precision_policy_sha256=precision,
            longform_policy_sha256=longform,
        )

    def validate_identity(self, identity: RuntimeIdentity) -> None:
        if identity != self.identity:
            raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "SAM runtime identity mismatch")
        for package, expected in self.PACKAGES.items():
            try:
                actual = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                raise CleanExecutionError(
                    ErrorCode.ENGINE_UNAVAILABLE, "SAM dependency missing"
                ) from None
            if actual != expected:
                raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "SAM dependency mismatch")
        torch = importlib.import_module("torch")
        if (
            torch.get_default_dtype() != torch.float32
            or torch.get_default_device().type != "cpu"
            or torch.is_autocast_enabled("cpu")
            or not torch.backends.mkldnn.enabled
            or torch.backends.mkldnn.deterministic
            or torch.are_deterministic_algorithms_enabled()
            or torch.get_float32_matmul_precision() != "highest"
        ):
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE, "SAM CPU precision policy mismatch"
            )
        if self.device == "cuda:0" and (
            torch.is_autocast_enabled("cuda")
            or torch.backends.cuda.matmul.allow_tf32
            or torch.backends.cudnn.allow_tf32
            or not torch.backends.cudnn.enabled
            or torch.backends.cudnn.benchmark
            or torch.backends.cudnn.deterministic
        ):
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE, "SAM CUDA precision policy mismatch"
            )

    def _place_modules(self, core, codec, guard: ResourceGuard) -> None:
        """Transfer only admitted audio modules, with a cap before any transfer."""
        if self.device == "cpu":
            return
        torch = importlib.import_module("torch")
        guard.check()
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE, "SAM requires one visible CUDA device"
            )
        total = torch.cuda.get_device_properties(0).total_memory
        cap = guard.budget.allocator_cap_bytes
        tensors = tuple(
            tensor
            for module in (core, codec)
            for tensor in (*module.parameters(), *module.buffers())
        )
        if any(t.device.type != "cpu" for t in tensors):
            raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "SAM CPU staging incomplete")
        required = sum(t.numel() * t.element_size() for t in tensors)
        if cap > total or required >= cap:
            raise CleanExecutionError(
                ErrorCode.RESOURCE_EXHAUSTED, "SAM weights exceed CUDA reservation"
            )
        # An allocator cap is not the aggregate NVML process-memory release limit.
        torch.cuda.set_per_process_memory_fraction(cap / total, device=0)
        del tensors
        for module in (core, codec):
            guard.check()
            module.to(device=self.device, dtype=torch.float32)
        if any(
            t.device != torch.device(self.device) or t.dtype != torch.float32
            for module in (core, codec)
            for t in (*module.parameters(), *module.buffers())
        ):
            raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "SAM CUDA transfer incomplete")
        guard.check()

    def open(self, guard: ResourceGuard) -> LoadedSamBackend:
        guard.check()
        self.validate_identity(self.identity)
        self.cache.get(self.prompt)
        manifest = json.loads(
            SamCoreBuilder._read(self.assets.optional_manifest, self.OPTIONAL_MANIFEST, guard)
        )
        if manifest["checkpoint_sha256"] != self.CHECKPOINT:
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE, "SAM manifest checkpoint mismatch"
            )
        core = codec = None
        try:
            core = SamCoreBuilder.build(self.assets.sam_source, self.assets.config, guard)
            codec = SamCodecBuilder.build(self.assets.codec_source, self.assets.config, guard)
            counts = SamCheckpointLoader.load(
                self.assets.checkpoint,
                sha256=self.CHECKPOINT,
                core=core.core,
                codec=codec.codec,
                optional_keys=frozenset(manifest["excluded_keys"]),
                guard=guard,
            )
            if counts != (247, 317):
                raise CleanExecutionError(
                    ErrorCode.ENGINE_UNAVAILABLE, "SAM module inventory mismatch"
                )
            if any(
                t.device.type != "cpu"
                for module in (core.core, codec.codec)
                for t in module.state_dict().values()
            ):
                raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "SAM CPU load incomplete")
            self._place_modules(core.core, codec.codec, guard)
            pipeline = SamSeparationPipeline(
                core.core, codec.codec, codec_tile_frames=self.codec_tile_frames, policy=self.solver
            )
            guard.check()
            return LoadedSamBackend(core, codec, pipeline, self.prompt, self.cache)
        except BaseException:
            try:
                if codec is not None:
                    codec.close()
            finally:
                if core is not None:
                    core.close()
            raise
