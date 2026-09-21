"""Offline, explicit DF3 loading for a dedicated cleaner worker process.

The release descriptor is trusted deployment configuration, never user input.
No downloader or permissive upstream checkpoint loader is invoked.
"""

import hashlib
import importlib
import importlib.metadata
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode, RuntimeIdentity


@dataclass(frozen=True)
class PinnedDeepFilterAssets:
    config_path: Path
    config_sha256: str
    checkpoint_path: Path
    checkpoint_sha256: str
    package_versions: tuple[tuple[str, str], ...]
    device: str

    @property
    def precision_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {"dtype": "float32", "autocast": False, "tf32": False},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    @property
    def runtime_sha256(self) -> str:
        # This identifies the adapter configuration, not a production image.
        # Image/source revisions still belong in the certification evidence.
        descriptor = {
            "loader_policy": "df3-offline-strict-v3",
            "fault_policy": "oom-or-cuda-runtime-fault-requires-process-restart",
            "config_sha256": self.config_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "packages": dict(self.package_versions),
            "device": self.device,
            "precision_sha256": self.precision_sha256,
            "postfilter": False,
        }
        return hashlib.sha256(
            json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def __post_init__(self):
        for digest in (self.config_sha256, self.checkpoint_sha256):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("asset digests must be SHA-256")
        versions = dict(self.package_versions)
        required = {"deepfilternet", "deepfilterlib", "torch", "torchaudio", "numpy"}
        if (
            not required.issubset(versions)
            or len(versions) != len(self.package_versions)
            or any(not version for version in versions.values())
        ):
            raise ValueError("DF3 runtime packages must be explicitly pinned")
        if self.device not in ("cpu", "cuda:0"):
            raise ValueError("unsupported DF3 device")

    def verify(self, guard: ResourceGuard) -> None:
        for path, expected in (
            (self.config_path, self.config_sha256),
            (self.checkpoint_path, self.checkpoint_sha256),
        ):
            guard.check()
            if path.is_symlink() or not path.is_file():
                raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "pinned DF3 asset missing")
            digest = hashlib.sha256()
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    guard.check()
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "pinned DF3 asset mismatch")
        for package, expected in self.package_versions:
            try:
                actual = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError as exc:
                raise CleanExecutionError(
                    ErrorCode.ENGINE_UNAVAILABLE, "DF3 dependency missing"
                ) from exc
            if actual != expected:
                raise CleanExecutionError(
                    ErrorCode.ENGINE_UNAVAILABLE, "DF3 dependency version mismatch"
                )


class PinnedDeepFilterFactory:
    # Upstream configuration is process-global. Keep one owner through inference.
    _runtime_lease = threading.Lock()
    _worker_faulted = threading.Event()

    def __init__(self, assets: PinnedDeepFilterAssets):
        self.assets = assets

    def identity(self, longform_policy_sha256: str) -> RuntimeIdentity:
        return RuntimeIdentity(
            engine="deepfilternet3",
            runtime_sha256=self.assets.runtime_sha256,
            checkpoint_sha256=self.assets.checkpoint_sha256,
            precision_policy_sha256=self.assets.precision_sha256,
            longform_policy_sha256=longform_policy_sha256,
        )

    def validate_identity(self, identity: RuntimeIdentity) -> None:
        self.assert_healthy()
        if identity != self.identity(identity.longform_policy_sha256):
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE, "DF3 assets/runtime identity mismatch"
            )

    @classmethod
    def assert_healthy(cls) -> None:
        if cls._worker_faulted.is_set():
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE,
                "DF3 worker requires process restart",
                worker_restart_required=True,
            )

    @staticmethod
    def verify_environment(config) -> None:
        # df.config consults uppercase environment variables before the INI.
        # Reject conflicts, even equal values; do not mutate the process environment.
        keys = {key.upper() for section in config.parser for key in config.parser[section]}
        if keys.intersection(os.environ):
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE, "environment overrides pinned DF3 configuration"
            )

    def open(self, guard: ResourceGuard):
        self.assert_healthy()
        self.assets.verify(guard)
        if not self._runtime_lease.acquire(blocking=False):
            raise CleanExecutionError(ErrorCode.RESOURCE_EXHAUSTED, "DF3 runtime already in use")
        try:
            return self._load(guard)
        except Exception as exc:
            self._runtime_lease.release()
            if isinstance(exc, CleanExecutionError):
                if exc.worker_restart_required:
                    self._worker_faulted.set()
                failure = CleanExecutionError(
                    exc.code, str(exc), worker_restart_required=exc.worker_restart_required
                )
            else:
                exhausted = isinstance(exc, (torch.cuda.OutOfMemoryError, MemoryError))
                if exhausted or (self.assets.device == "cuda:0" and isinstance(exc, RuntimeError)):
                    self._worker_faulted.set()
                failure = CleanExecutionError(
                    ErrorCode.RESOURCE_EXHAUSTED if exhausted else ErrorCode.ENGINE_UNAVAILABLE,
                    "pinned DF3 runtime failed to load",
                    worker_restart_required=self._worker_faulted.is_set(),
                )
        except BaseException:
            self._runtime_lease.release()
            raise
        # Raise outside the handler: retaining the native exception's traceback
        # can keep partially loaded CUDA tensors alive through its frame locals.
        raise failure

    def _load(self, guard: ResourceGuard):
        # Dedicated worker policy; do not inherit another caller's TF32 mode.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        config = importlib.import_module("df.config").config
        config.load(
            str(self.assets.config_path),
            config_must_exist=True,
            allow_defaults=False,
            allow_reload=True,
        )
        self.verify_environment(config)
        if config("MODEL", section="train") != "deepfilternet3":
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE, "checkpoint is not DeepFilterNet3"
            )
        # Stage required model weights on CPU, irrespective of GPU availability.
        config.set("DEVICE", "cpu", str, "train")
        config.set("MASK_PF", False, bool, "deepfilternet")
        module = importlib.import_module("df.deepfilternet3")
        params = module.ModelParams()
        if params.sr != 48000:
            raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "DF3 model rate mismatch")
        state_args = dict(
            sr=params.sr,
            fft_size=params.fft_size,
            hop_size=params.hop_size,
            nb_bands=params.nb_erb,
            min_nb_erb_freqs=params.min_nb_freqs,
        )
        state_factory = importlib.import_module("libdf").DF
        model = module.init_model(state_factory(**state_args), run_df=True, train_mask=True)
        weights = torch.load(self.assets.checkpoint_path, map_location="cpu", weights_only=True)
        # No key deletion, legacy renaming or strict=False; incompatible assets fail.
        model.load_state_dict(weights, strict=True)
        del weights
        for value in model.state_dict().values():
            if not torch.isfinite(value).all():
                raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "non-finite DF3 weights")
        model.eval()
        guard.check()
        if self.assets.device == "cuda:0":
            if not torch.cuda.is_available():
                raise CleanExecutionError(
                    ErrorCode.ENGINE_UNAVAILABLE, "configured CUDA device unavailable"
                )
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(
                min(1.0, guard.budget.allocator_cap_bytes / total), device=0
            )
        model = model.to(device=self.assets.device, dtype=torch.float32)
        config.set("DEVICE", self.assets.device, str, "train")
        enhance = importlib.import_module("df.enhance").enhance
        guard.check()
        return LoadedDeepFilter(
            model,
            state_factory,
            state_args,
            enhance,
            self._runtime_lease,
            guard,
            config,
            self.assets.device.split(":")[0],
        )


class LoadedDeepFilter:
    def __init__(
        self,
        model,
        state_factory,
        state_args: dict[str, int],
        enhance,
        lease: threading.Lock,
        guard: ResourceGuard,
        config,
        device_type: str,
    ):
        self.model = model
        self.state_factory = state_factory
        self.state_args = state_args
        self.enhance_function = enhance
        self.lease = lease
        self.guard = guard
        self.config = config
        self.device_type = device_type
        self.closed = False

    def enhance(self, samples: np.ndarray, attenuation_limit_db: int) -> np.ndarray:
        self.guard.check()
        PinnedDeepFilterFactory.verify_environment(self.config)
        if self.closed or attenuation_limit_db not in (12, 18, 24):
            raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "invalid DF3 inference session")
        PinnedDeepFilterFactory.assert_healthy()
        output = None
        try:
            # Contextual windows start from clean filter/STFT state as well as h0.
            state = self.state_factory(**self.state_args)
            with (
                torch.inference_mode(),
                torch.autocast(device_type=self.device_type, enabled=False),
            ):
                output = self.enhance_function(
                    self.model,
                    state,
                    torch.from_numpy(samples),
                    pad=True,
                    atten_lim_db=attenuation_limit_db,
                )
            self.guard.check()
            return output.detach().cpu().numpy().astype(np.float32, copy=False)
        except CleanExecutionError as exc:
            if exc.worker_restart_required:
                PinnedDeepFilterFactory._worker_faulted.set()
            failure = CleanExecutionError(
                exc.code, str(exc), worker_restart_required=exc.worker_restart_required
            )
        except (MemoryError, RuntimeError, ValueError) as exc:
            exhausted = isinstance(exc, (torch.cuda.OutOfMemoryError, MemoryError))
            if exhausted or (self.device_type == "cuda" and isinstance(exc, RuntimeError)):
                PinnedDeepFilterFactory._worker_faulted.set()
            code = ErrorCode.RESOURCE_EXHAUSTED if exhausted else ErrorCode.PROCESS_FAILED
            failure = CleanExecutionError(
                code,
                "DF3 inference failed",
                worker_restart_required=PinnedDeepFilterFactory._worker_faulted.is_set(),
            )
        # Close the failed model and discard a possible CUDA output before raising
        # the sanitized error outside the handler. No retry or quality fallback.
        output = None
        self.close()
        raise failure

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            try:
                self.model = None
            finally:
                self.lease.release()
