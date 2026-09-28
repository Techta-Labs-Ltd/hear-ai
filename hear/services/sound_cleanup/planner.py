"""Bounded event plans: detector output never authorises removing wanted speech."""

import hashlib
import math
from dataclasses import asdict, dataclass

import numpy as np

from hear.contracts.sound_cleanup import SoundCleanupOptions
from hear.services.sound_cleanup.analysis import SoundAnalysis


@dataclass
class SoundRegion:
    start: int
    end: int
    kind: str
    origin: str
    score: float | None = None
    confirmed_no_speech: bool = False
    outcome: str = "pending"
    reason: str = ""
    method: str | None = None
    reduction_db: float | None = None
    separation_checks: dict | None = None

    def report(self) -> dict:
        data = asdict(self)
        data["start_seconds"] = self.start / 48000
        data["end_seconds"] = self.end / 48000
        data["event_id"] = hashlib.sha256(
            f"{self.start}:{self.end}:{self.kind}:{self.origin}".encode()
        ).hexdigest()[:24]
        return data


class SoundRepairPlanner:
    VERSION = "sound-cleanup-local-v1"
    MAX_EVENTS = 128
    MAX_REPAIRED_SECONDS = 60
    MAX_COVERAGE = 0.25

    @staticmethod
    def runs(mask: np.ndarray):
        padded = np.pad(mask.astype(np.int8), (1, 1))
        return zip(
            np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1), strict=True
        )

    @classmethod
    def plan(
        cls, analysis: SoundAnalysis, options: SoundCleanupOptions
    ) -> tuple[list[SoundRegion], bool]:
        count = math.ceil(analysis.frames / analysis.step_frames)
        arrays = (
            analysis.speech,
            analysis.source_rms,
            analysis.clean_rms,
            *analysis.events.values(),
        )
        if (
            analysis.step_frames != 1536
            or analysis.channels not in (1, 2)
            or analysis.frames < 1
            or analysis.protected.shape != (count,)
            or analysis.speech.shape != (count,)
        ):
            raise ValueError("invalid_sound_analysis_grid")
        if analysis.source_rms.shape != (count, analysis.channels) or analysis.clean_rms.shape != (
            count,
            analysis.channels,
        ):
            raise ValueError("invalid_sound_analysis_channel_layout")
        if any(not np.isfinite(a).all() for a in arrays) or any(
            a.shape != (count,) for a in analysis.events.values()
        ):
            raise ValueError("invalid_sound_analysis_values")
        selected = [
            SoundRegion(
                round(r.start_seconds * 48000),
                round(r.end_seconds * 48000),
                r.kind,
                "selected",
                confirmed_no_speech=r.confirmed_no_speech,
            )
            for r in options.regions
        ]
        if any(r.end > analysis.frames or r.start >= r.end for r in selected):
            raise ValueError("selected_sound_region_exceeds_recording")
        truncated = False
        targets = set(options.targets) | ({"cough"} if options.remove_coughs else set())
        automatic: list[SoundRegion] = []
        if options.auto_detect:
            for kind in sorted(targets):
                scores = analysis.events.get(kind)
                if scores is None:
                    raise ValueError("requested_sound_detector_unavailable")
                for lo, hi in cls.runs(scores >= 0.45):
                    score = float(np.max(scores[lo:hi]))
                    if score < 0.70 or (hi - lo) * analysis.step_frames < 0.08 * 48000:
                        continue
                    start = max(0, int(lo) * analysis.step_frames - 3072)
                    end = min(analysis.frames, int(hi) * analysis.step_frames + 4608)
                    if any(start < r.end and end > r.start for r in selected):
                        continue
                    automatic.append(SoundRegion(start, end, kind, "detected", score))
                    if len(automatic) >= cls.MAX_EVENTS * 4:
                        truncated = True
                        break
        merged: list[SoundRegion] = []
        for item in sorted(automatic, key=lambda r: (r.start, r.end, r.kind)):
            if merged and item.start <= merged[-1].end:
                previous = merged[-1]
                previous.end = max(previous.end, item.end)
                previous.kind = "+".join(sorted(set(previous.kind.split("+")) | {item.kind}))
                previous.score = max(previous.score or 0, item.score or 0)
            else:
                merged.append(item)
        remaining = cls.MAX_EVENTS - len(selected)
        truncated |= len(merged) > remaining
        regions = sorted(selected + merged[:remaining], key=lambda r: r.start)
        planned_frames = 0
        for region in regions:
            lo = region.start // analysis.step_frames
            hi = min(len(analysis.speech), math.ceil(region.end / analysis.step_frames))
            duration = region.end - region.start
            if duration < 8:
                region.outcome, region.reason = "needs_review", "event_too_short_for_local_repair"
            elif duration > 8 * 48000:
                region.outcome, region.reason = "needs_review", "event_too_long_for_local_repair"
            elif np.any(analysis.protected[lo:hi]) and not region.confirmed_no_speech:
                region.outcome, region.reason = "needs_review", "speech_or_wanted_content_overlap"
                if options.preview_overlaps and region.origin == "selected":
                    if planned_frames + duration > min(
                        cls.MAX_REPAIRED_SECONDS * 48000, analysis.frames * cls.MAX_COVERAGE
                    ):
                        region.reason = "repair_coverage_limit"
                    else:
                        planned_frames += duration
            elif float(np.max(analysis.clean_rms[lo:hi], initial=0)) < 10 ** (-60 / 20):
                region.outcome, region.reason = "unchanged", "already_below_processing_floor"
            elif planned_frames + duration > min(
                cls.MAX_REPAIRED_SECONDS * 48000, analysis.frames * cls.MAX_COVERAGE
            ):
                region.outcome, region.reason = "needs_review", "repair_coverage_limit"
            else:
                planned_frames += duration
        return regions, truncated
