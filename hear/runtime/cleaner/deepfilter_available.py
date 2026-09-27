from __future__ import annotations

import hashlib
import importlib.metadata
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from hear.runtime.cleaner.deepfilter_loader import PinnedDeepFilterAssets, PinnedDeepFilterFactory
from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.services.magic_clean.contracts import CleanPlan
from hear.services.magic_clean.engines.deepfilter import ContextualPolicy, DeepFilterEngine


class DeepFilterNetCleaner:
    profile = "natural"
    engine = "deepfilternet3"
    config_sha256 = "0a926b0471793d7ba7446b07a8bdc10eafa5c9e3b93de4d65496e2cbcacc40d3"
    checkpoint_sha256 = "23b92884f63ccf54bb026014604625ab231657b6480df65db4095c4c171e6003"

    def __init__(
        self,
        config_path: Path,
        model_directory: Path,
        budget: ResourceBudget,
        *,
        device: str = "cuda:0",
    ) -> None:
        checkpoint = model_directory / "checkpoints" / "model_120.ckpt.best"
        packages = tuple(
            (name, importlib.metadata.version(name))
            for name in ("deepfilternet", "deepfilterlib", "torch", "torchaudio", "numpy")
        )
        assets = PinnedDeepFilterAssets(
            config_path,
            self.config_sha256,
            checkpoint,
            self.checkpoint_sha256,
            packages,
            device,
        )
        self._assets = assets
        self._budget = budget
        self._policy = ContextualPolicy(480_000, 48_000)
        self._factory = PinnedDeepFilterFactory(assets)
        self._identity = self._factory.identity(self._policy.digest)
        self._engine = DeepFilterEngine(self._identity, self._factory, self._policy)

    def is_ready(self) -> bool:
        try:
            self._verify_file(self._assets.config_path, self.config_sha256)
            self._verify_file(self._assets.checkpoint_path, self.checkpoint_sha256)
            self._factory.validate_identity(self._identity)
            return True
        except Exception:
            return False

    def clean(
        self,
        source: Path,
        target: Path,
        workspace: Path,
        options: dict,
        deadline: datetime,
        timeout_seconds: float,
    ) -> None:
        remaining = max(1.0, (deadline - datetime.now(UTC)).total_seconds())
        guard = ResourceGuard(
            self._budget,
            workspace,
            time.monotonic() + min(remaining, timeout_seconds),
            threading.Event(),
        )
        guard.bind_deadline(deadline)
        prepared = workspace / "deepfilter_input.wav"
        processed = workspace / "deepfilter_output.wav"
        self._run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                str(source),
                "-vn",
                "-ar",
                "48000",
                "-c:a",
                "pcm_f32le",
                str(prepared),
            ],
            timeout_seconds,
        )
        attenuation = int(options.get("attenuation_limit_db", 24))
        if attenuation not in (12, 18, 24):
            raise ValueError("invalid_deepfilter_attenuation_limit")
        plan = CleanPlan(
            profile="natural",
            profile_version="deepfilternet3-v1",
            catalogue_sha256=hashlib.sha256(b"deepfilternet3-v1").hexdigest(),
            runtime=self._identity,
            attenuation_limit_db=attenuation,
            prompt_sha256=None,
            channel_policy="preserve",
            mono_acknowledged=False,
            adjust_loudness=False,
            match_comparison_loudness=True,
            shorten_pauses=False,
            seed=0,
        )
        session = self._engine.open_session(plan, guard)
        try:
            session.process(prepared, processed, plan, guard)
        finally:
            session.close()
        self._run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                str(processed),
                "-c:a",
                "flac",
                str(target),
            ],
            timeout_seconds,
        )

    def close(self) -> None:
        self._engine.close()

    @staticmethod
    def _verify_file(path: Path, expected: str) -> None:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise RuntimeError("deepfilter_asset_digest_mismatch")

    @staticmethod
    def _run(command: list[str], timeout_seconds: float) -> None:
        subprocess.run(command, capture_output=True, check=True, timeout=timeout_seconds)
