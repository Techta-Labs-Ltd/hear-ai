from __future__ import annotations

import hashlib
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.runtime.cleaner.sam_loader import PinnedSamAssets, PinnedSamBaseFactory
from hear.services.magic_clean.contracts import CleanPlan
from hear.services.magic_clean.engines.sam_audio import SamEngine


class SamAudioCleaner:
    profile = "sam_audio"
    engine = "sam_audio_base"
    text_hashes = (
        ("config.json", "46dd7cb62d29c81fb551e0ef1ea274c24a46ba441eeb948897706252933df033"),
        ("tokenizer.json", "d2acde0d8d71dd30a711834b07781b9c89feaac33fd332f60507699282740066"),
        ("spiece.model", "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86"),
        ("model.safetensors", "a90903540cc02cbeb7ff9f823f1a80eb778c7e22426a0e620b01c77a5ec8f5b4"),
    )
    ranker_sha256 = "e02951eae3c9955db546c50086059e6457188ec39446858f2bddbcb4b56b1cb3"
    span_hashes = (
        ("config.json", "382227d331004428a954209d29609046d06db755b347c9232b27271d921f1126"),
        (
            "model.safetensors",
            "cb1b7d596f1765e6fe21707f7c78989b06e08bc5692ffa88868ad266cce65660",
        ),
        (
            "preprocessor_config.json",
            "d68bb68c371d05defe1f07dc25fbf211e865368dd4702b8bb85fd3eb518df20d",
        ),
        (
            "special_tokens_map.json",
            "ea97ecdbcc73713039d8d64dbb05e3689495c96657fbd9a18f5bed381be81049",
        ),
        (
            "tokenizer.json",
            "9fd55248d51d33976b324fc11592e28071da7d41e0e9401dfb7082e30574b7b1",
        ),
        (
            "tokenizer_config.json",
            "3cd2017ff46d0a527e5d39cae39272eccfa1f19bb9f89b05d166aab2e38354e2",
        ),
    )

    def __init__(
        self,
        model_directory: Path,
        text_directory: Path,
        budget: ResourceBudget,
        *,
        device: str = "cuda:0",
    ) -> None:
        assets = PinnedSamAssets(
            model_directory / "config.json",
            model_directory / "checkpoint.pt",
            text_directory,
            self.text_hashes,
            model_directory.parent / "laion-clap" / "630k-best.pt",
            self.ranker_sha256,
            model_directory.parent / "pe-a-frame-large",
            self.span_hashes,
            model_directory.parents[1] / ".cache" / "huggingface",
        )
        self._assets = assets
        self._budget = budget
        self._factory = PinnedSamBaseFactory(
            assets,
            device=device,
            text_encoder_identity=assets.text_identity,
        )
        self._identity = self._factory.identity
        self._engine = SamEngine(self._identity, self._factory)

    def is_ready(self) -> bool:
        try:
            self._verify_file(self._assets.config, PinnedSamBaseFactory.CONFIG_SHA256)
            self._verify_file(self._assets.checkpoint, PinnedSamBaseFactory.CHECKPOINT)
            for filename, digest in self.text_hashes:
                self._verify_file(self._assets.text_directory / filename, digest)
            self._verify_file(self._assets.ranker_checkpoint, self._assets.ranker_sha256)
            for filename, digest in self.span_hashes:
                self._verify_file(self._assets.span_directory / filename, digest)
            if not self._assets.dependency_cache_directory.is_dir():
                return False
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
        prepared = workspace / "sam_audio_input.wav"
        processed = workspace / "sam_audio_output.wav"
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
                "-ac",
                "1",
                "-ar",
                "48000",
                "-c:a",
                "pcm_f32le",
                str(prepared),
            ],
            timeout_seconds,
        )
        prompt = str(options.get("prompt") or "").strip().lower()
        action = str(options.get("action") or "remove").strip().lower()
        prompt_mode = str(options.get("prompt_mode") or "ambient").strip().lower()
        if (
            not prompt
            or action not in {"isolate", "remove"}
            or prompt_mode not in {"ambient", "event"}
        ):
            raise ValueError("invalid_sam_audio_prompt")
        plan = CleanPlan(
            profile="sam_audio",
            profile_version="sam-audio-base-text-v1",
            catalogue_sha256=hashlib.sha256(b"sam-audio-base-text-v1").hexdigest(),
            runtime=self._identity,
            attenuation_limit_db=None,
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            prompt_text=prompt,
            prompt_action=action,
            prompt_mode=prompt_mode,
            channel_policy="mono",
            mono_acknowledged=True,
            adjust_loudness=False,
            match_comparison_loudness=True,
            shorten_pauses=False,
            seed=int(options.get("seed", 0)),
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
            raise RuntimeError("sam_audio_asset_digest_mismatch")

    @staticmethod
    def _run(command: list[str], timeout_seconds: float) -> None:
        subprocess.run(command, capture_output=True, check=True, timeout=timeout_seconds)
