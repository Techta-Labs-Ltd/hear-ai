import hashlib
import importlib
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

from hear.runtime.cleaner.asset_probe import PinnedAssetProbe
from hear.runtime.cleaner.resampling import AudioResampler
from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.runtime.cleaner.sam_official import SamOfficialPipeline
from hear.runtime.cleaner.subprocesses import CancellableProcessRunner
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode, RuntimeIdentity


@dataclass(frozen=True)
class PinnedSamAssets:
    config: Path
    checkpoint: Path
    text_directory: Path
    text_hashes: tuple[tuple[str, str], ...]
    ranker_checkpoint: Path
    ranker_sha256: str
    span_directory: Path
    span_hashes: tuple[tuple[str, str], ...]
    dependency_cache_directory: Path

    @property
    def text_identity(self) -> str:
        return PinnedSamFactory._digest(
            {
                "version": "meta-sam-audio-local-t5-v1",
                "files": dict(self.text_hashes),
            }
        )

    @property
    def quality_identity(self) -> str:
        return PinnedSamFactory._digest(
            {
                "version": "meta-sam-audio-clap-pe-v1",
                "ranker": self.ranker_sha256,
                "span": dict(self.span_hashes),
            }
        )

    def verify(self, guard: ResourceGuard) -> None:
        self._verify_file(self.config, PinnedSamFactory.CONFIG_SHA256, guard)
        self._verify_file(self.checkpoint, PinnedSamFactory.CHECKPOINT, guard)
        for filename, digest in self.text_hashes:
            self._verify_file(self.text_directory / filename, digest, guard)
        self._verify_file(self.ranker_checkpoint, self.ranker_sha256, guard)
        for filename, digest in self.span_hashes:
            self._verify_file(self.span_directory / filename, digest, guard)
        if not self.dependency_cache_directory.is_dir():
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE,
                "SAM Audio dependency cache is unavailable",
            )

    @staticmethod
    def _verify_file(path: Path, expected: str, guard: ResourceGuard) -> None:
        PinnedAssetProbe.sha256(path, expected, check=guard.check)

    def __post_init__(self) -> None:
        expected = {"config.json", "tokenizer.json", "spiece.model", "model.safetensors"}
        values = dict(self.text_hashes)
        if set(values) != expected or len(values) != len(self.text_hashes):
            raise ValueError("SAM text assets must be completely pinned")
        if any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in values.values()
        ):
            raise ValueError("SAM text asset digests must be SHA-256")
        expected_span = {
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        }
        span_values = dict(self.span_hashes)
        if set(span_values) != expected_span or len(span_values) != len(self.span_hashes):
            raise ValueError("SAM span assets must be completely pinned")
        quality_hashes = (self.ranker_sha256, *span_values.values())
        if any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in quality_hashes
        ):
            raise ValueError("SAM quality asset digests must be SHA-256")


class LoadedSamBackend:
    def __init__(self, model, processor, pipeline):
        self.model = model
        self.processor = processor
        self.pipeline = pipeline
        self.closed = False

    def borrow(self):
        if self.closed or self.pipeline is None:
            raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "SAM cached runtime is closed")
        return BorrowedSamBackend(self)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.pipeline = self.processor = self.model = None


class BorrowedSamBackend:
    def __init__(self, owner: LoadedSamBackend):
        self.owner = owner
        self.closed = False

    @property
    def pipeline(self):
        if self.closed or self.owner.closed or self.owner.pipeline is None:
            raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "SAM attempt backend is closed")
        return self.owner.pipeline

    def close(self):
        self.closed = True


class PinnedSamFactory:
    ENGINE = "sam_audio_base"
    CHECKPOINT = "b5f3e29ea7a9e80e90a00da495a8aafe890571f371c4bfb88c052c65a5636839"
    CONFIG_SHA256 = "b99a0ee6296edaeb8d355d41d365b33faa94b40af00b2c34d643a43617b10fb2"
    META_SOURCE_COMMIT = "bb4c6999d2677c7402360e426afc01ddfad6dce0"
    PACKAGES = {
        "sam-audio": "0.1.0",
        "einops": "0.8.2",
        "safetensors": "0.8.0",
        "sentencepiece": "0.2.2",
        "torch": "2.8.0+cu128",
        "torchaudio": "2.8.0+cu128",
        "tokenizers": "0.22.2",
        "transformers": "4.57.6",
        "numpy": "1.26.4",
        "laion-clap": "1.1.6",
        "perception-models": "1.0.0",
        "timm": "1.0.30",
    }

    def __init__(
        self,
        assets: PinnedSamAssets,
        *,
        device: str = "cpu",
        text_encoder_identity: str,
    ):
        if device not in ("cpu", "cuda:0"):
            raise ValueError("unsupported SAM device")
        if device == "cuda:0":
            torch = importlib.import_module("torch")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.assets = assets
        self.device = device
        self.text_encoder_identity = text_encoder_identity
        self._cache_lock = threading.Lock()
        self._loaded: LoadedSamBackend | None = None

    @staticmethod
    def _digest(value) -> str:
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def identity(self) -> RuntimeIdentity:
        precision = self._digest(
            {
                "device": self.device,
                "dtype": "float32",
                "autocast": False,
                "tf32": False,
            }
        )
        runtime_descriptor = {
            "loader": "meta-sam-audio-official-api-v1",
            "source_commit": self.META_SOURCE_COMMIT,
            "checkpoint": self.CHECKPOINT,
            "config": self.CONFIG_SHA256,
            "packages": self.PACKAGES,
            "audio_only": "vision-features-zeroed-by-upstream-no-video-path",
            "quality_modules": "official-clap-ranker-and-pe-span-predictor",
            "quality_assets": self.assets.quality_identity,
            "ambient_reranking_candidates": SamOfficialPipeline.AMBIENT_RERANKING_CANDIDATES,
            "event_reranking_candidates": SamOfficialPipeline.EVENT_RERANKING_CANDIDATES,
            "precision": precision,
            "channel_policy": SamOfficialPipeline.CHANNEL_POLICY,
            "dual_mono_min_correlation": SamOfficialPipeline.DUAL_MONO_MIN_CORRELATION,
            "resampling": AudioResampler.POLICY,
            "subprocess_runner": CancellableProcessRunner.POLICY,
            "subprocess_shutdown_seconds": (
                CancellableProcessRunner.DEFAULT_SHUTDOWN_TIMEOUT_SECONDS
            ),
            "longform_policy": SamOfficialPipeline.POLICY,
            "chunk_seconds": SamOfficialPipeline.CHUNK_SECONDS,
            "overlap_seconds": SamOfficialPipeline.OVERLAP_SECONDS,
        }
        runtime_descriptor["text_encoder_identity"] = self.text_encoder_identity
        return RuntimeIdentity(
            engine=self.ENGINE,
            checkpoint_sha256=self.CHECKPOINT,
            runtime_sha256=self._digest(runtime_descriptor),
            precision_policy_sha256=precision,
            longform_policy_sha256=self._digest(
                {
                    "policy": SamOfficialPipeline.POLICY,
                    "channel_policy": SamOfficialPipeline.CHANNEL_POLICY,
                    "dual_mono_min_correlation": (SamOfficialPipeline.DUAL_MONO_MIN_CORRELATION),
                    "resampling": AudioResampler.POLICY,
                    "subprocess_runner": CancellableProcessRunner.POLICY,
                    "subprocess_shutdown_seconds": (
                        CancellableProcessRunner.DEFAULT_SHUTDOWN_TIMEOUT_SECONDS
                    ),
                    "chunk_seconds": SamOfficialPipeline.CHUNK_SECONDS,
                    "overlap_seconds": SamOfficialPipeline.OVERLAP_SECONDS,
                    "ambient_reranking_candidates": (
                        SamOfficialPipeline.AMBIENT_RERANKING_CANDIDATES
                    ),
                    "event_reranking_candidates": (SamOfficialPipeline.EVENT_RERANKING_CANDIDATES),
                }
            ),
        )

    def validate_identity(self, identity: RuntimeIdentity) -> None:
        if identity != self.identity:
            raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "SAM runtime identity mismatch")
        PinnedAssetProbe.packages(self.PACKAGES)
        torch = importlib.import_module("torch")
        if torch.get_default_dtype() != torch.float32 or torch.is_autocast_enabled("cpu"):
            raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "SAM FP32 policy mismatch")
        if self.device == "cuda:0" and (
            not torch.cuda.is_available()
            or torch.cuda.device_count() != 1
            or torch.backends.cuda.matmul.allow_tf32
            or torch.backends.cudnn.allow_tf32
        ):
            raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "SAM CUDA policy mismatch")

    def open(self, guard: ResourceGuard):
        guard.check()
        self.validate_identity(self.identity)
        with self._cache_lock:
            if self._loaded is None or self._loaded.closed:
                self._loaded = self._load(guard)
            return self._loaded.borrow()

    def close(self) -> None:
        with self._cache_lock:
            if self._loaded is not None:
                self._loaded.close()
                self._loaded = None

    def _load(self, guard: ResourceGuard) -> LoadedSamBackend:
        guard.check()
        self.validate_identity(self.identity)
        if type(self) is PinnedSamFactory:
            self.assets.verify(guard)
        else:
            self.assets._verify_file(self.assets.config, self.CONFIG_SHA256, guard)
            self.assets._verify_file(self.assets.checkpoint, self.CHECKPOINT, guard)
            for filename, digest in self.assets.text_hashes:
                self.assets._verify_file(self.assets.text_directory / filename, digest, guard)
            self.assets._verify_file(
                self.assets.ranker_checkpoint,
                self.assets.ranker_sha256,
                guard,
            )
            for filename, digest in self.assets.span_hashes:
                self.assets._verify_file(self.assets.span_directory / filename, digest, guard)
        try:
            os.environ["HF_HOME"] = str(self.assets.dependency_cache_directory)
            os.environ["HF_HUB_CACHE"] = str(self.assets.dependency_cache_directory / "hub")
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
            torch = importlib.import_module("torch")
            sam_audio = importlib.import_module("sam_audio")
            SAMAudio = sam_audio.SAMAudio
            SAMAudioProcessor = sam_audio.SAMAudioProcessor

            model_directory = self.assets.config.parent
            model = SAMAudio.from_pretrained(
                str(model_directory),
                local_files_only=True,
                text_encoder={"name": str(self.assets.text_directory)},
                visual_ranker=None,
                text_ranker={
                    "kind": "clap",
                    "checkpoint": str(self.assets.ranker_checkpoint),
                },
                span_predictor=str(self.assets.span_directory),
            )
            processor = SAMAudioProcessor.from_pretrained(str(model_directory))

            vision_dim = model.vision_encoder.dim
            del model.vision_encoder
            model._vision_encoder_dim = vision_dim

            def audio_only_video_features(_video, audio_features):
                batch, frames, _ = audio_features.shape
                return audio_features.new_zeros(batch, model._vision_encoder_dim, frames)

            model._get_video_features = audio_only_video_features
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            model.eval().to(device=self.device, dtype=torch.float32)
            if self.device == "cuda:0":
                model.span_predictor.to(device="cpu", dtype=torch.float32)
            guard.check()
            pipeline = SamOfficialPipeline(model, processor)
            return LoadedSamBackend(model, processor, pipeline)
        except CleanExecutionError:
            raise
        except Exception as exc:
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE,
                f"official SAM Audio initialization failed ({type(exc).__name__})",
            ) from None


PinnedSamBaseFactory = PinnedSamFactory
