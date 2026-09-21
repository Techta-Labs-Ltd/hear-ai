"""Single Cleaner v2 mastering path with exact-file validation.

Only linear gain is applied. Loudness targets yield to +6 dB and true-peak
constraints. Codec correction always starts with the float engine output.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.runtime.cleaner.subprocesses import CancellableProcessRunner
from hear.services.magic_clean.contracts import CleanExecutionError, CleanPlan, ErrorCode


@dataclass(frozen=True)
class LoudnessMeasurement:
    integrated_lufs: float | None
    unavailable_reason: str | None
    true_peak_dbtp: float | None


@dataclass(frozen=True)
class MasteredAudio:
    master: Path
    delivery: Path
    gain_db: float
    processing_rate: int
    delivery_rate: int
    frames: int
    channels: int
    master_measurement: LoudnessMeasurement
    delivery_measurement: LoudnessMeasurement
    bit_depth: int = 24
    dither_policy: str = "none"


class AudioMasteringService:
    DELIVERY_RATE = 48000
    MAX_CORRECTIONS = 2

    def __init__(self, runner: CancellableProcessRunner):
        self.runner = runner

    @staticmethod
    def _input_args(path: Path) -> list[str]:
        return [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-nostats",
            "-v",
            "info",
            "-threads",
            "1",
            "-filter_threads",
            "1",
            "-protocol_whitelist",
            "file,pipe",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-vn",
            "-sn",
            "-dn",
            "-map_metadata",
            "-1",
        ]

    def measure(self, path: Path, guard: ResourceGuard, duration: float) -> LoudnessMeasurement:
        diagnostic = self.runner.run(
            self._input_args(path)
            + ["-af", "loudnorm=I=-19:TP=-1:LRA=11:print_format=json", "-f", "null", "-"],
            guard,
        ).decode("utf-8", errors="replace")
        start = diagnostic.rfind("{")
        end = diagnostic.rfind("}")
        try:
            data = json.loads(diagnostic[start : end + 1])
            loudness = float(data["input_i"])
            peak = float(data["input_tp"])
        except (ValueError, KeyError, TypeError) as exc:
            raise CleanExecutionError(
                ErrorCode.INVALID_AUDIO, "missing loudness measurement"
            ) from exc
        if math.isnan(loudness) or math.isnan(peak) or peak == math.inf or loudness == math.inf:
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "invalid loudness measurement")
        reason = None
        if duration < 0.4:
            reason = "too_short"
        elif loudness == -math.inf:
            reason = "below_measurement_gate"
        return LoudnessMeasurement(
            None if reason else loudness, reason, None if peak == -math.inf else peak
        )

    @staticmethod
    def scan(
        path: Path,
        guard: ResourceGuard,
        *,
        rate: int,
        channels: int,
        frames: int,
        tolerance: int = 0,
    ) -> None:
        actual = 0
        try:
            with sf.SoundFile(path) as audio:
                if audio.samplerate != rate or audio.channels != channels:
                    raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "encoded layout mismatch")
                while True:
                    guard.check()
                    block = audio.read(32768, dtype="float64", always_2d=True)
                    if not len(block):
                        break
                    actual += len(block)
                    if actual > frames + tolerance or not np.isfinite(block).all():
                        raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "invalid encoded PCM")
                if abs(actual - frames) > tolerance:
                    raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "encoded timeline mismatch")
        except (RuntimeError, ValueError) as exc:
            if isinstance(exc, CleanExecutionError):
                raise
            raise CleanExecutionError(
                ErrorCode.INVALID_AUDIO, "encoded audio decode failed"
            ) from exc

    def master(self, source: Path, plan: CleanPlan, guard: ResourceGuard) -> MasteredAudio:
        try:
            return self._master(source, plan, guard)
        except (OSError, RuntimeError, ValueError) as exc:
            if isinstance(exc, CleanExecutionError):
                raise
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "audio mastering failed") from exc

    def _master(self, source: Path, plan: CleanPlan, guard: ResourceGuard) -> MasteredAudio:
        guard.check()
        source = source.resolve()
        if not source.is_relative_to(guard.workspace.resolve()):
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "mastering source outside workspace")
        with sf.SoundFile(source) as audio:
            rate, channels, frames = audio.samplerate, audio.channels, audio.frames
            if channels not in (1, 2) or frames < 1 or audio.subtype not in ("FLOAT", "DOUBLE"):
                raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "mastering requires float PCM")
        guard.preflight_pcm(frames, channels, copies=3, output_bytes=source.stat().st_size + 8192)
        self.scan(source, guard, rate=rate, channels=channels, frames=frames)
        measured = self.measure(source, guard, frames / rate)
        gain = 0.0
        if plan.adjust_loudness and measured.integrated_lufs is not None:
            target = -19 if channels == 1 else -16
            gain = min(6.0, target - measured.integrated_lufs)
        if measured.true_peak_dbtp is not None:
            gain = min(gain, -1.2 - measured.true_peak_dbtp)
        created = []
        try:
            for correction in range(self.MAX_CORRECTIONS + 1):
                master = guard.workspace / f"master-{correction}.flac"
                delivery = guard.workspace / f"delivery-{correction}.mp3"
                if master.exists() or delivery.exists():
                    raise CleanExecutionError(
                        ErrorCode.ARTIFACT_CONFLICT, "master output already exists"
                    )
                created.extend((master, delivery))
                self.runner.run(
                    self._input_args(source)
                    + [
                        "-n",
                        "-af",
                        f"volume={gain:.8f}dB:precision=double",
                        "-c:a",
                        "flac",
                        "-sample_fmt",
                        "s32",
                        "-bits_per_raw_sample",
                        "24",
                        "-ar",
                        str(rate),
                        "-threads",
                        "1",
                        str(master),
                    ],
                    guard,
                )
                self.scan(master, guard, rate=rate, channels=channels, frames=frames)
                with sf.SoundFile(master) as audio:
                    if audio.subtype != "PCM_24":
                        raise CleanExecutionError(
                            ErrorCode.INVALID_AUDIO, "master bit depth mismatch"
                        )
                self.runner.run(
                    self._input_args(master)
                    + [
                        "-n",
                        "-c:a",
                        "libmp3lame",
                        "-b:a",
                        "128k" if channels == 1 else "192k",
                        "-ar",
                        str(self.DELIVERY_RATE),
                        "-threads",
                        "1",
                        str(delivery),
                    ],
                    guard,
                )
                expected_delivery_frames = round(frames * self.DELIVERY_RATE / rate)
                self.scan(
                    delivery,
                    guard,
                    rate=self.DELIVERY_RATE,
                    channels=channels,
                    frames=expected_delivery_frames,
                    tolerance=1440,
                )
                master_stats = self.measure(master, guard, frames / rate)
                delivery_stats = self.measure(delivery, guard, frames / rate)
                peaks = [
                    value
                    for value in (master_stats.true_peak_dbtp, delivery_stats.true_peak_dbtp)
                    if value is not None
                ]
                peak = max(peaks, default=-math.inf)
                if peak <= -1:
                    guard.check()
                    return MasteredAudio(
                        master,
                        delivery,
                        gain,
                        rate,
                        self.DELIVERY_RATE,
                        frames,
                        channels,
                        master_stats,
                        delivery_stats,
                    )
                # Re-render from the original float intermediate, never from MP3.
                gain -= peak + 1.2
                master.unlink()
                delivery.unlink()
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "delivered true-peak gate failed")
        except BaseException:
            for path in created:
                path.unlink(missing_ok=True)
            raise
