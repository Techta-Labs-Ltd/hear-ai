import configparser
import gc
import hashlib
import importlib
import json
import logging
import os
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from hear.runtime.cleaner.asset_probe import PinnedAssetProbe
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

        descriptor = {
            "loader_policy": "deepfilternet-official-init-df-v1",
            "fault_policy": "oom-or-cuda-runtime-fault-requires-process-restart",
            "config_sha256": self.config_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "packages": dict(self.package_versions),
            "device": self.device,
            "precision_sha256": self.precision_sha256,
            "postfilter": "per_plan",
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
            PinnedAssetProbe.sha256(path, expected, check=guard.check)
        PinnedAssetProbe.packages(self.package_versions)


class PinnedDeepFilterFactory:
    _runtime_lease = threading.Lock()
    _worker_faulted = threading.Event()

    def __init__(
        self,
        assets: PinnedDeepFilterAssets,
        *,
        idle_seconds: float = 300,
        eviction_enabled: bool = True,
    ):
        if idle_seconds <= 0:
            raise ValueError("invalid_deepfilter_idle_ttl")
        self.assets = assets
        self._cache_lock = threading.Lock()
        self._loaded = None
        self._idle_seconds = idle_seconds
        self._eviction_enabled = eviction_enabled
        self._idle_timer: threading.Timer | None = None

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
    def verify_environment(parser: configparser.ConfigParser) -> None:

        keys = {key.upper() for section in parser for key in parser[section]}
        if keys.intersection(os.environ):
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE, "environment overrides pinned DF3 configuration"
            )

    def open(self, guard: ResourceGuard):
        self.assert_healthy()
        guard.check()
        with self._cache_lock:
            self._cancel_idle_locked()
            loaded = self._loaded
            if loaded is not None and not loaded.closed:
                return loaded.borrow(guard, self._borrow_released)
            self._loaded = None
            self.assets.verify(guard)
            if not self._runtime_lease.acquire(blocking=False):
                raise CleanExecutionError(
                    ErrorCode.RESOURCE_EXHAUSTED, "DF3 runtime already in use"
                )
            try:
                loaded = self._load(guard)
            except Exception as exc:
                self._runtime_lease.release()
                if isinstance(exc, CleanExecutionError):
                    if exc.worker_restart_required:
                        self._worker_faulted.set()
                    failure = CleanExecutionError(
                        exc.code,
                        str(exc),
                        worker_restart_required=exc.worker_restart_required,
                    )
                else:
                    exhausted = isinstance(exc, (torch.cuda.OutOfMemoryError, MemoryError))
                    if exhausted or (
                        self.assets.device == "cuda:0" and isinstance(exc, RuntimeError)
                    ):
                        self._worker_faulted.set()
                    failure = CleanExecutionError(
                        ErrorCode.RESOURCE_EXHAUSTED if exhausted else ErrorCode.ENGINE_UNAVAILABLE,
                        "pinned DF3 runtime failed to load",
                        worker_restart_required=self._worker_faulted.is_set(),
                    )
            except BaseException:
                self._runtime_lease.release()
                raise
            else:
                self._loaded = loaded
                logging.getLogger(__name__).info("gpu_model_loaded name=deepfilternet3")
                return loaded.borrow(guard, self._borrow_released)

        raise failure

    def _cancel_idle_locked(self) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _borrow_released(self) -> None:
        with self._cache_lock:
            self._cancel_idle_locked()
            if not self._eviction_enabled or self._loaded is None:
                return
            timer = threading.Timer(self._idle_seconds, self._evict_idle)
            timer.daemon = True
            self._idle_timer = timer
            timer.start()

    def _evict_idle(self) -> None:
        with self._cache_lock:
            self._idle_timer = None
            loaded = self._loaded
            self._loaded = None
        if loaded is not None:
            loaded.close()
            logging.getLogger(__name__).info("gpu_model_evicted name=deepfilternet3")

    def evict_now(self) -> bool:
        with self._cache_lock:
            self._cancel_idle_locked()
            loaded = self._loaded
            self._loaded = None
        if loaded is None:
            return False
        loaded.close()
        return True

    def close(self) -> None:
        with self._cache_lock:
            self._cancel_idle_locked()
            loaded = self._loaded
            self._loaded = None
        if loaded is not None:
            loaded.close()

    def _load(self, guard: ResourceGuard):

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if self.assets.device == "cuda:0" and not torch.cuda.is_available():
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE, "configured CUDA device unavailable"
            )
        parser = configparser.ConfigParser()
        config_bytes = self.assets.config_path.read_bytes()
        if hashlib.sha256(config_bytes).hexdigest() != self.assets.config_sha256:
            raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "pinned DF3 config changed")
        parser.read_string(config_bytes.decode("utf-8"))
        self.verify_environment(parser)
        if parser.get("train", "model", fallback="").lower() != "deepfilternet3":
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE, "checkpoint is not DeepFilterNet3"
            )

        model_directory = tempfile.TemporaryDirectory(prefix="hear-df3-model-")
        try:
            model_root = Path(model_directory.name) / "DeepFilterNet3"
            checkpoints = model_root / "checkpoints"
            checkpoints.mkdir(parents=True)
            staged_config = model_root / "config.ini"
            parser.set("train", "device", self.assets.device)
            with staged_config.open("w") as target:
                parser.write(target)
            staged_checkpoint = checkpoints / "model_120.ckpt.best"
            shutil.copyfile(self.assets.checkpoint_path, staged_checkpoint)
            if hashlib.sha256(staged_checkpoint.read_bytes()).hexdigest() != (
                self.assets.checkpoint_sha256
            ):
                raise CleanExecutionError(
                    ErrorCode.ENGINE_UNAVAILABLE, "pinned DF3 checkpoint changed"
                )
            guard.check()
            enhance_module = importlib.import_module("df.enhance")
            enhance = enhance_module.enhance
            init_df = enhance_module.init_df

            initialized = init_df(
                model_base_dir=str(model_root),
                post_filter=False,
                log_file=None,
                config_allow_defaults=False,
                epoch="best",
            )

            if not isinstance(initialized, tuple) or len(initialized) not in (3, 4):
                raise CleanExecutionError(
                    ErrorCode.ENGINE_UNAVAILABLE, "unexpected DeepFilterNet init_df result"
                )
            model, df_state, suffix = initialized[:3]
            epoch = initialized[3] if len(initialized) == 4 else 120
            if suffix != "DeepFilterNet3":
                raise CleanExecutionError(
                    ErrorCode.ENGINE_UNAVAILABLE, "unexpected DeepFilterNet model identity"
                )
            if epoch != 120 or next(model.parameters()).dtype != torch.float32:
                raise CleanExecutionError(
                    ErrorCode.ENGINE_UNAVAILABLE, "unexpected DF3 checkpoint or precision"
                )
            if next(model.parameters()).device.type != self.assets.device.split(":")[0]:
                raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "DF3 device mismatch")
            guard.check()
        except BaseException:
            model_directory.cleanup()
            raise
        return LoadedDeepFilterRuntime(
            model,
            df_state,
            enhance,
            self._runtime_lease,
            model_directory,
            self.assets.device.split(":")[0],
        )


class LoadedDeepFilterRuntime:
    def __init__(
        self,
        model,
        df_state,
        enhance,
        lease: threading.Lock,
        model_directory,
        device_type: str,
    ):
        self.model = model
        self.df_state = df_state
        self.enhance_function = enhance
        self.lease = lease
        self.model_directory = model_directory
        self.device_type = device_type
        self.closed = False

    def borrow(self, guard: ResourceGuard, on_close=lambda: None):
        guard.check()
        if self.closed or self.model is None:
            raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "DF3 cached runtime is closed")
        return LoadedDeepFilter(self, guard, on_close)

    def enhance(
        self,
        samples: np.ndarray,
        attenuation_limit_db: int,
        post_filter: bool,
        guard: ResourceGuard,
    ) -> np.ndarray:
        guard.check()
        if (
            self.closed
            or self.model is None
            or type(attenuation_limit_db) is not int
            or not 6 <= attenuation_limit_db <= 60
        ):
            raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "invalid DF3 inference session")
        PinnedDeepFilterFactory.assert_healthy()
        output = None
        try:
            # The DF3 module reads this flag at forward time; the runtime lease
            # serialises calls, so each plan sets its own value.
            self.model.post_filter = bool(post_filter)
            with (
                torch.inference_mode(),
                torch.autocast(device_type=self.device_type, enabled=False),
            ):
                output = self.enhance_function(
                    self.model,
                    self.df_state,
                    torch.from_numpy(samples),
                    pad=True,
                    atten_lim_db=attenuation_limit_db,
                )
            guard.check()
            return output.detach().cpu().numpy().astype(np.float32, copy=False)
        except CleanExecutionError as exc:
            if exc.worker_restart_required:
                PinnedDeepFilterFactory._worker_faulted.set()
            failure = CleanExecutionError(
                exc.code,
                str(exc),
                worker_restart_required=exc.worker_restart_required,
            )
        except (MemoryError, RuntimeError, ValueError) as exc:
            exhausted = isinstance(exc, (torch.cuda.OutOfMemoryError, MemoryError))
            if exhausted or (self.device_type == "cuda" and isinstance(exc, RuntimeError)):
                PinnedDeepFilterFactory._worker_faulted.set()
            failure = CleanExecutionError(
                ErrorCode.RESOURCE_EXHAUSTED if exhausted else ErrorCode.PROCESS_FAILED,
                "DF3 inference failed",
                worker_restart_required=PinnedDeepFilterFactory._worker_faulted.is_set(),
            )
        output = None
        self.close()
        raise failure

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            try:
                self.model = None
                self.df_state = None
            finally:
                try:
                    self.model_directory.cleanup()
                finally:
                    if self.lease.locked():
                        self.lease.release()
                    gc.collect()
                    if self.device_type == "cuda" and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()


class LoadedDeepFilter:
    def __init__(self, runtime: LoadedDeepFilterRuntime, guard: ResourceGuard, on_close):
        self.runtime = runtime
        self.guard = guard
        self._on_close = on_close
        self.closed = False

    @property
    def model(self):
        return self.runtime.model

    def enhance(
        self, samples: np.ndarray, attenuation_limit_db: int, post_filter: bool
    ) -> np.ndarray:
        self.guard.check()
        if self.closed:
            raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "DF3 attempt session is closed")
        try:
            return self.runtime.enhance(samples, attenuation_limit_db, post_filter, self.guard)
        except BaseException:
            self.closed = True
            raise

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._on_close()
