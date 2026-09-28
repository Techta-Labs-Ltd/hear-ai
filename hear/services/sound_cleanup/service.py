"""Local repair with per-event outcomes and sample-preserving rollback."""

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import soundfile as sf

from hear.contracts.sound_cleanup import SoundCleanupOptions
from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode
from hear.services.magic_clean.mastering import AudioMasteringService
from hear.services.sound_cleanup.analysis import SoundAnalyser, SoundAnalysis
from hear.services.sound_cleanup.planner import SoundRegion, SoundRepairPlanner
from hear.services.sound_cleanup.preview_integrity import PreviewIntegrity
from hear.services.sound_cleanup.separator import EventSeparator


class SoundCleanupService:
    def __init__(self, analyser: SoundAnalyser, separator: EventSeparator | None = None):
        self.analyser = analyser
        self.separator = separator

    @staticmethod
    def blend_weight(frames: int) -> np.ndarray:
        ramp = min(2400, max(2, frames // 4))
        weight = np.ones(frames, dtype=np.float64)
        fade = np.sin(np.linspace(0, np.pi / 2, ramp)) ** 2
        weight[:ramp], weight[-ramp:] = fade, fade[::-1]
        return weight

    @staticmethod
    def _room_tone(audio, region: SoundRegion, evidence: SoundAnalysis, regions):
        rms = np.max(evidence.clean_rms, axis=1)
        local = rms[region.start // 1536 : math.ceil(region.end / 1536)]
        floor = min(0.003, float(np.max(local, initial=0)) / 8)
        safe = (~evidence.protected) & (evidence.speech < 0.03) & (rms < floor)
        for scores in evidence.events.values():
            safe &= scores < 0.20
        for event in regions:
            safe[event.start // 1536 : math.ceil(event.end / 1536)] = False
        candidates = []
        for lo, hi in SoundRepairPlanner.runs(safe):
            if (hi - lo) * 1536 < 24000:
                continue
            if (
                min(abs(int(lo) * 1536 - region.end), abs(int(hi) * 1536 - region.start))
                > 5 * 48000
            ):
                continue
            values = rms[lo:hi]
            if float(np.quantile(values, 0.9)) > max(float(np.quantile(values, 0.1)) * 2.5, 1e-7):
                continue
            candidates.append((abs(int(lo) * 1536 - region.start), int(lo), int(hi)))
        if not candidates:
            return None
        _, lo, hi = min(candidates)
        audio.seek(lo * 1536)
        grain = audio.read(min((hi - lo) * 1536, 48000), dtype="float64", always_2d=True)
        if len(grain) < 24000 or not np.isfinite(grain).all():
            return None
        frames = region.end - region.start
        overlap = 1200
        tiled = np.zeros((frames + len(grain), evidence.channels), dtype=np.float64)
        weights = np.zeros(frames + len(grain), dtype=np.float64)
        window = np.ones(len(grain))
        window[:overlap] = np.linspace(0.001, 1, overlap)
        window[-overlap:] = window[:overlap][::-1]
        for start in range(0, frames, len(grain) - overlap):
            tiled[start : start + len(grain)] += grain * window[:, None]
            weights[start : start + len(grain)] += window
        return tiled[:frames] / np.maximum(weights[:frames, None], 1e-12)

    @classmethod
    def candidate(cls, audio, region, evidence, regions):
        audio.seek(region.start)
        before = audio.read(region.end - region.start, dtype="float64", always_2d=True)
        if len(before) < 8 or not np.isfinite(before).all():
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "invalid_sound_region")
        if region.kind == "click" and len(before) <= 288:
            replacement = np.linspace(before[0], before[-1], len(before))
            method = "short_impulse_interpolation"
        else:
            replacement = cls._room_tone(audio, region, evidence, regions)
            method = "local_room_tone"
            if replacement is None:
                replacement = before * 10 ** (-18 / 20)
                method = "bounded_event_attenuation"
        weight = cls.blend_weight(len(before))[:, None]
        after = before * (1 - weight) + replacement * weight
        margin = min(2400, len(before) // 4)
        measured_before = before[margin : len(before) - margin]
        measured_after = after[margin : len(before) - margin]
        power_before = float(np.mean(measured_before**2))
        power_after = float(np.mean(measured_after**2))
        reduction = 10 * math.log10(max(power_before, 1e-20) / max(power_after, 1e-20))
        if not np.isfinite(after).all() or np.max(np.abs(after)) > np.max(np.abs(before)) + 1e-8:
            return None, method, reduction, "candidate_integrity_failed"
        if reduction < 3:
            return None, method, reduction, "insufficient_event_energy_reduction"
        after[0], after[-1] = before[0], before[-1]
        return after.astype("float32"), method, reduction, ""

    def run(
        self,
        original: Path,
        baseline: Path,
        target: Path,
        options: SoundCleanupOptions,
        guard: ResourceGuard,
    ) -> dict:
        guard.check()
        if options.preview_overlaps and self.separator is None:
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE, "overlap_separator_not_provisioned"
            )
        if not options.enabled:
            raise ValueError("sound_cleanup_disabled")
        for path in (original, baseline, target):
            if path.is_symlink() or not path.resolve().is_relative_to(guard.workspace.resolve()):
                raise CleanExecutionError(
                    ErrorCode.INVALID_AUDIO, "sound_cleanup_path_outside_workspace"
                )
        if target.exists():
            raise CleanExecutionError(ErrorCode.ARTIFACT_CONFLICT, "sound_cleanup_output_exists")
        with sf.SoundFile(baseline) as audio:
            if any(round(r.end_seconds * 48000) > audio.frames for r in options.regions):
                raise CleanExecutionError(
                    ErrorCode.INVALID_AUDIO, "selected_sound_region_exceeds_recording"
                )
        evidence = self.analyser.analyse(
            original, baseline, guard, detect_events=options.auto_detect
        )
        regions, truncated = SoundRepairPlanner.plan(evidence, options)
        guard.preflight_pcm(evidence.frames, evidence.channels, copies=1, output_bytes=8192)
        temporary = target.with_name(target.stem + ".incomplete.wav")
        if temporary.exists():
            raise CleanExecutionError(
                ErrorCode.ARTIFACT_CONFLICT, "sound_cleanup_incomplete_output_exists"
            )
        try:
            with temporary.open("xb") as stream, sf.SoundFile(baseline) as source:
                with sf.SoundFile(
                    stream,
                    "w",
                    samplerate=48000,
                    channels=evidence.channels,
                    format="WAV",
                    subtype="FLOAT",
                ) as output:
                    while True:
                        guard.check_scratch()
                        block = source.read(32768, dtype="float32", always_2d=True)
                        if not len(block):
                            break
                        output.write(block)
            with sf.SoundFile(baseline) as source, sf.SoundFile(temporary, "r+") as output:
                for region in regions:
                    guard.check()
                    if (
                        options.preview_overlaps
                        and region.origin == "selected"
                        and region.reason == "speech_or_wanted_content_overlap"
                    ):
                        assert self.separator is not None
                        try:
                            candidate, reason, metrics = self.separator.propose(
                                source, region, evidence, self.analyser, guard
                            )
                        finally:
                            self.separator.close()
                        region.method = "audiosep_selected_overlap_preview"
                        region.separation_checks = metrics
                        if candidate is None:
                            region.outcome, region.reason = "rejected", reason
                            continue
                        source.seek(region.start)
                        before = source.read(
                            region.end - region.start, dtype="float32", always_2d=True
                        )
                        weight = self.blend_weight(len(before))[:, None]
                        blended = (before * (1 - weight) + candidate * weight).astype("float32")
                        reason, blend_metrics = PreviewIntegrity.assess(before, blended)
                        region.separation_checks["blended_artifact_check"] = blend_metrics
                        if reason:
                            region.outcome, region.reason = "rejected", reason
                            continue
                        output.seek(region.start)
                        output.write(blended)
                        region.outcome, region.reason = (
                            "preview",
                            "manual_listening_review_required",
                        )
                        continue
                    if region.outcome != "pending":
                        continue
                    candidate, method, reduction, reason = self.candidate(
                        source, region, evidence, regions
                    )
                    region.method, region.reduction_db = method, round(reduction, 4)
                    if candidate is None:
                        region.outcome, region.reason = "rejected", reason
                        continue
                    output.seek(region.start)
                    output.write(candidate)
                    region.outcome = "reduced"
                    region.reason = (
                        "user_confirmed_no_speech"
                        if region.confirmed_no_speech
                        else "speech_checks_passed_review_required"
                    )
            AudioMasteringService.scan(
                temporary, guard, rate=48000, channels=evidence.channels, frames=evidence.frames
            )
            self.verify_untouched(baseline, temporary, regions, guard)
            guard.check()
            os.link(temporary, target)
            temporary.unlink()
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        previews = sum(r.outcome == "preview" for r in regions)
        repaired = sum(r.outcome == "reduced" for r in regions)
        unresolved = sum(r.outcome in ("needs_review", "rejected") for r in regions)
        recipe = {
            "version": SoundRepairPlanner.VERSION,
            "assets": evidence.model_digest,
            "separator_policy": EventSeparator.POLICY_VERSION if self.separator else None,
            "separator_manifest": getattr(self.separator, "digest", None),
            "preview_integrity_policy": PreviewIntegrity.POLICY,
            "options": options.model_dump(mode="json"),
        }
        return {
            "enabled": True,
            "version": SoundRepairPlanner.VERSION,
            "recipe_sha256": hashlib.sha256(
                json.dumps(recipe, sort_keys=True).encode()
            ).hexdigest(),
            "analysis_assets_sha256": evidence.model_digest,
            "status": "partial"
            if unresolved or truncated or previews
            else ("reduced" if repaired else "no_changes"),
            "events": [r.report() for r in regions],
            "repaired_count": repaired,
            "unresolved_count": unresolved,
            "preview_count": previews,
            "events_truncated": truncated,
            "audio_changed": repaired > 0 or previews > 0,
            "sample_rate": 48000,
            "frames": evidence.frames,
            "channels": evidence.channels,
            "outside_repair_regions_unchanged": True,
            "requires_approval": True,
            "word_retention_certified": False,
            "overlap_separation_available": self.separator is not None,
            "detection_thresholds_calibrated": False,
            "speech_protection": "original_and_deepfilter_all_channels",
            "detector_peak_scores": {
                name: round(float(np.max(scores, initial=0)), 4)
                for name, scores in evidence.events.items()
            },
            "speech_protected_fraction": round(float(np.mean(evidence.protected)), 4),
            "explicit_speech_overrides": sum(
                r.confirmed_no_speech and r.outcome == "reduced" for r in regions
            ),
        }

    @staticmethod
    def verify_untouched(baseline: Path, repaired: Path, regions, guard):
        active = [r for r in regions if r.outcome in ("reduced", "preview")]
        offset = 0
        with sf.SoundFile(baseline) as a, sf.SoundFile(repaired) as b:
            while True:
                guard.check()
                before = a.read(32768, dtype="float32", always_2d=True)
                after = b.read(32768, dtype="float32", always_2d=True)
                if before.shape != after.shape or not np.isfinite(after).all():
                    raise CleanExecutionError(
                        ErrorCode.INVALID_AUDIO, "sound_cleanup_output_mismatch"
                    )
                if not len(before):
                    break
                editable = np.zeros(len(before), dtype=bool)
                for region in active:
                    lo = max(0, region.start - offset)
                    hi = min(len(before), region.end - offset)
                    if hi > lo:
                        editable[lo:hi] = True
                if not np.array_equal(before[~editable], after[~editable]):
                    raise CleanExecutionError(
                        ErrorCode.INVALID_AUDIO, "audio_changed_outside_repair_regions"
                    )
                offset += len(before)
