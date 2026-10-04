from __future__ import annotations

import hashlib
import importlib.metadata
import os
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from hear.contracts.cleaning import PROFILE_VERSION, CleaningProfiles, MagicCleanProfile
from hear.contracts.sound_cleanup import SoundCleanupOptions
from hear.runtime.cleaner.asset_probe import PinnedAssetSet
from hear.runtime.cleaner.deepfilter_loader import PinnedDeepFilterAssets, PinnedDeepFilterFactory
from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.runtime.cleaner.subprocesses import CancellableProcessRunner
from hear.services.magic_clean.contracts import CleanExecutionError, CleanPlan, ErrorCode
from hear.services.magic_clean.engines.deepfilter import ContextualPolicy, DeepFilterEngine
from hear.services.magic_clean.mastering import AudioMasteringService
from hear.services.magic_clean.profile_dsp import ProfileDspService, TrimResult
from hear.services.magic_clean.quality import AudioQualityGate
from hear.services.sound_cleanup.background import BackgroundCleanup
from hear.services.sound_cleanup.service import SoundCleanupService


class DeepFilterNetCleaner:
    profile = "natural"
    engine = "deepfilternet3"
    supported_profiles = frozenset(profile.value for profile in MagicCleanProfile)
    config_sha256 = "0a926b0471793d7ba7446b07a8bdc10eafa5c9e3b93de4d65496e2cbcacc40d3"
    checkpoint_sha256 = "23b92884f63ccf54bb026014604625ab231657b6480df65db4095c4c171e6003"

    def __init__(
        self,
        config_path: Path,
        model_directory: Path,
        budget: ResourceBudget,
        *,
        device: str = "cuda:0",
        sound_cleanup_service: SoundCleanupService | None = None,
        idle_seconds: float = 300,
        eviction_enabled: bool = True,
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
        self._pinned_assets = PinnedAssetSet(
            (
                (assets.config_path, self.config_sha256, 1024 * 1024),
                (assets.checkpoint_path, self.checkpoint_sha256, None),
            )
        )
        self._sound_cleanup = sound_cleanup_service
        self._budget = budget
        self._policy = ContextualPolicy(480_000, 48_000)
        self._factory = PinnedDeepFilterFactory(
            assets,
            idle_seconds=idle_seconds,
            eviction_enabled=eviction_enabled,
        )
        self._identity = self._factory.identity(self._policy.digest)
        self._engine = DeepFilterEngine(self._identity, self._factory, self._policy)
        self._runner = CancellableProcessRunner()
        self._dsp = ProfileDspService(self._runner)
        self._mastering = AudioMasteringService(self._runner)

    @property
    def sound_cleanup_available(self) -> bool:
        return self._sound_cleanup is not None

    @property
    def overlap_preview_available(self) -> bool:
        return self._sound_cleanup is not None and self._sound_cleanup.separator is not None

    def is_ready(self) -> bool:
        try:
            self._pinned_assets.verify()
            self._factory.validate_identity(self._identity)
            return self._dsp.is_ready()
        except Exception:
            return False

    def warmup(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hear-deepfilter-warmup-") as raw:
            guard = ResourceGuard(
                self._budget, Path(raw), time.monotonic() + 300, threading.Event()
            )
            self._factory.open(guard).close()

    def clean(
        self,
        source: Path,
        workspace: Path,
        options: dict,
        deadline: datetime,
        timeout_seconds: float,
        *,
        cancelled: threading.Event | None = None,
        progress: Callable[[str, float], None] | None = None,
    ) -> dict:
        started = time.perf_counter()
        timings: dict[str, float] = {}
        phase_started = started
        options = CleaningProfiles.validate(options)
        sound_options = SoundCleanupOptions.model_validate(options.get("sound_cleanup", {}))
        if (
            sound_options.enabled or options.get("reduce_stationary_noise")
        ) and self._sound_cleanup is None:
            raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "sound_cleanup_not_provisioned")
        remaining = (deadline - datetime.now(UTC)).total_seconds()
        guard = ResourceGuard(
            self._budget,
            workspace,
            time.monotonic() + min(remaining, timeout_seconds),
            cancelled if cancelled is not None else threading.Event(),
        )
        guard.bind_deadline(deadline)
        guard.check_scratch()
        if source.stat().st_size > self._budget.max_input_bytes:
            raise CleanExecutionError(ErrorCode.RESOURCE_EXHAUSTED, "input exceeds byte limit")
        prepared = workspace / "deepfilter_input.wav"
        conditioned = workspace / "deepfilter_conditioned.wav"
        processed = workspace / "deepfilter_output.wav"
        finished = workspace / "profile_output.wav"
        self._dsp.render(source, prepared, [], guard, decode=True)
        frames, channels = self._dsp.validate(prepared, guard)
        guard.preflight_pcm(frames, channels, copies=5, output_bytes=frames * channels * 4)
        timings["decode_and_validate_seconds"] = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        original_measurement = self._mastering.measure(prepared, guard, frames / 48000)
        timings["input_loudness_measurement_seconds"] = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        preparation_filters = self._dsp.preparation_filters(options)
        model_input = prepared
        if preparation_filters:
            self._dsp.render(prepared, conditioned, preparation_filters, guard)
            model_input = conditioned
        # Preserve the established pinned DeepFilterNet engine contract. User
        # profiles select DSP and attenuation, never a substitute/fallback model.
        plan = CleanPlan(
            profile="natural",
            profile_version=PROFILE_VERSION,
            catalogue_sha256=hashlib.sha256(PROFILE_VERSION.encode()).hexdigest(),
            runtime=self._identity,
            attenuation_limit_db=options["attenuation_limit_db"],
            post_filter=options["post_filter"],
            prompt_sha256=None,
            channel_policy="preserve",
            mono_acknowledged=False,
            adjust_loudness=options["auto_level"],
            match_comparison_loudness=True,
            shorten_pauses=False,
            seed=0,
        )
        timings["preprocessing_seconds"] = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        if progress:
            progress("denoising", 30)
        session = self._engine.open_session(plan, guard)
        try:
            session.process(model_input, processed, plan, guard)
        finally:
            session.close()
        timings["denoising_seconds"] = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        AudioMasteringService.scan(processed, guard, rate=48000, channels=channels, frames=frames)
        quality = AudioQualityGate().evaluate(model_input, processed, plan, guard)
        speech_report = {"status": "analyser_not_provisioned"}
        if self._sound_cleanup is not None:
            speech_report = self.speech_preservation(
                self._sound_cleanup.analyser, model_input, processed, guard
            )
        timings["quality_checks_seconds"] = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        sound_report = {"enabled": False, "status": "not_requested"}
        if sound_options.enabled:
            if progress:
                progress("sound_cleanup", 55)
            repaired = workspace / "sound_repaired.wav"
            assert self._sound_cleanup is not None
            sound_report = self._sound_cleanup.run(
                prepared, processed, repaired, sound_options, guard
            )
            processed = repaired
        background_report = {"enabled": False, "status": "not_requested"}
        if options.get("reduce_stationary_noise"):
            assert self._sound_cleanup is not None
            if progress:
                progress("background_cleanup", 65)
            background_evidence = self._sound_cleanup.analyser.analyse(
                prepared, processed, guard, detect_events=False
            )
            background_output = workspace / "background_cleaned.wav"
            background_report = {
                "enabled": True,
                **BackgroundCleanup().render(
                    processed, background_output, background_evidence, guard
                ),
            }
            background_probability = self._sound_cleanup.analyser.speech_probability(
                background_output, guard
            )
            anchors = background_evidence.speech >= 0.8
            lost = anchors & (background_probability < 0.1)
            background_report["lost_high_confidence_speech_frames"] = int(lost.sum())
            if int(lost.sum()) > 1:
                background_report["status"] = "rejected_speech_activity_loss"
                background_output.unlink(missing_ok=True)
            else:
                processed = background_output
        timings["optional_event_and_background_cleanup_seconds"] = (
            time.perf_counter() - phase_started
        )
        phase_started = time.perf_counter()
        finish_filters = self._dsp.finishing_filters(options)
        master_input = processed
        if finish_filters:
            self._dsp.render(processed, finished, finish_filters, guard)
            AudioMasteringService.scan(
                finished, guard, rate=48000, channels=channels, frames=frames
            )
            master_input = finished
        trim = TrimResult(0, frames, frames)
        if options["trim_silence"]:
            trimmed = workspace / "trimmed_output.wav"
            trim = self._dsp.trim_edges(master_input, trimmed, guard)
            master_input = trimmed
        timings["finishing_and_trim_seconds"] = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        if progress:
            progress("mastering", 75)
        mastered = self._mastering.master(master_input, plan, guard)
        timings["mastering_and_export_seconds"] = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        delivery = workspace / "delivery_audio.mp3"
        guard.check()
        if delivery.exists():
            raise CleanExecutionError(ErrorCode.ARTIFACT_CONFLICT, "cleaning output already exists")
        os.replace(mastered.delivery, delivery)
        warnings = list(quality.warning_codes)
        if speech_report.get("status") != "analyser_not_provisioned":
            # The VAD comparison supersedes the energy gate's missing-analyser notice.
            warnings = [code for code in warnings if code != "speech_activity_unavailable"]
        if speech_report.get("status") == "review_required":
            warnings.append("possible_speech_loss")
        if sound_report.get("status") == "partial":
            warnings.append("sound_cleanup_some_events_need_review")
        target_lufs = (-19.0 if channels == 1 else -16.0) if options["auto_level"] else None
        measured_lufs = mastered.delivery_measurement.integrated_lufs
        if target_lufs is not None and (
            measured_lufs is None or abs(measured_lufs - target_lufs) > 1.0
        ):
            warnings.append("loudness_target_limited_by_headroom_or_measurement_gate")
        if not options["auto_level"] and mastered.gain_db < 0:
            warnings.append("linear_attenuation_applied_for_peak_safety")
        return {
            "engine": self.engine,
            "performance": {
                **{k: round(v, 6) for k, v in timings.items()},
                "total_seconds": round(time.perf_counter() - started, 6),
            },
            "profile": options["profile"],
            "profile_version": PROFILE_VERSION,
            "sound_cleanup": sound_report,
            "background_cleanup": background_report,
            "effective_options": {k: v for k, v in options.items() if k != "cleaner_ticket"},
            "input_measurement": asdict(original_measurement),
            "loudness_meter": {
                "method": "ffmpeg_ebur128_true_peak",
                "loudness_resolution_lu": 0.1,
                "true_peak_rounding_safety_db": 0.05,
            },
            "delivery_measurement": asdict(mastered.delivery_measurement),
            "target_lufs": target_lufs,
            "gain_db": mastered.gain_db,
            "channels": channels,
            "sample_rate": 48000,
            "duration_seconds": mastered.frames / 48000,
            "timeline": asdict(trim),
            "warnings": warnings,
            "technical_validation": "passed",
            "perceptual_review_required": True,
            "content_validation": quality.model_dump(mode="json"),
            "speech_preservation": speech_report,
        }

    @staticmethod
    def speech_preservation(analyser, source: Path, processed: Path, guard) -> dict:
        """Fail when confidently voiced frames vanish; energy checks cannot see this."""
        before = analyser.speech_probability(source, guard)
        after = analyser.speech_probability(processed, guard)
        anchors = before >= 0.8
        lost = anchors & (after < 0.1)
        anchor_count = int(anchors.sum())
        lost_count = int(lost.sum())
        allowed = max(2, anchor_count // 100)
        if lost_count > allowed:
            raise CleanExecutionError(
                ErrorCode.INVALID_AUDIO,
                f"speech_activity_lost:{lost_count}_of_{anchor_count}_voiced_frames",
            )
        return {
            "status": "review_required" if lost_count else "passed",
            "voiced_frames": anchor_count,
            "lost_voiced_frames": lost_count,
            "allowed_lost_frames": allowed,
            "step_frames": int(getattr(analyser, "STEP", 1536)),
        }

    def close(self) -> None:
        self._engine.close()
        if self._sound_cleanup is not None:
            self._sound_cleanup.close()
