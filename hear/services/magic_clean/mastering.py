"""Single mastering path with exact-file validation.

Only linear gain is applied. Loudness targets yield to +6 dB and true-peak
constraints. The delivery MP3 is rendered straight from the float engine output
and any codec overshoot correction re-renders from that float source.

Long recordings are the common case, so the float master is metered in parallel
ranges, encoded as frame-aligned MP3 pieces in parallel and verified by decoding
those pieces again; see `loudness` and `mp3` for the two algorithms.
"""

import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import soundfile as sf

from hear.runtime.cleaner.pool import WorkerPool
from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.runtime.cleaner.subprocesses import CancellableProcessRunner
from hear.services.magic_clean.contracts import CleanExecutionError, CleanPlan, ErrorCode
from hear.services.magic_clean.loudness import BlockMeasurement, KWeightedMeter
from hear.services.magic_clean.mp3 import (
    FRAME_SAMPLES,
    EncodeTask,
    Mp3Frames,
    ParallelMp3,
    VerifyTask,
)


@dataclass(frozen=True)
class MasteringSettings:
    adjust_loudness: bool = False


@dataclass(frozen=True)
class LoudnessMeasurement:
    integrated_lufs: float | None
    unavailable_reason: str | None
    true_peak_dbtp: float | None


@dataclass(frozen=True)
class MasteredAudio:
    delivery: Path
    gain_db: float
    processing_rate: int
    delivery_rate: int
    frames: int
    channels: int
    delivery_measurement: LoudnessMeasurement


@dataclass(frozen=True)
class MeasureTask:
    path: str
    rate: int
    start: int
    end: int
    warmup_frames: int


class AudioMasteringService:
    DELIVERY_RATE = 48000
    MAX_CORRECTIONS = 2
    MEASURE_RANGE_FRAMES = 2**23
    MEASURE_WARMUP_FRAMES = 48000
    # Whole-frame padding plus the decoder's own warm-up.
    DELIVERY_FRAME_TOLERANCE = 1440 + FRAME_SAMPLES

    def __init__(self, runner: CancellableProcessRunner, *, workers: int | None = None):
        self.runner = runner
        self.workers = workers or min(8, os.cpu_count() or 1)

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
        """Reference ffmpeg meter; slow, kept for cross-checks and short clips."""
        diagnostic = self.runner.run(
            self._input_args(path)
            + ["-af", "ebur128=peak=true:framelog=verbose", "-f", "null", "-"],
            guard,
        ).decode("utf-8", errors="replace")
        summary = diagnostic.rsplit("Summary:", 1)
        if len(summary) != 2:
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "missing loudness measurement")
        integrated = re.search(r"Integrated loudness:\s*I:\s*([-+0-9.infna]+) LUFS", summary[1])
        true_peak = re.search(r"True peak:\s*Peak:\s*([-+0-9.infna]+) dBFS", summary[1])
        if integrated is None or true_peak is None:
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "missing loudness measurement")
        loudness = float(integrated.group(1))
        peak = float(true_peak.group(1))
        if math.isnan(loudness) or math.isnan(peak) or peak == math.inf or loudness == math.inf:
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "invalid loudness measurement")
        reason = "too_short" if duration < 0.4 else None
        if reason is None and loudness <= -70.0:
            reason = "below_measurement_gate"
        return LoudnessMeasurement(
            None if reason else loudness,
            reason,
            None if peak == -math.inf else round(peak + 0.05, 6),
        )

    @staticmethod
    def measure_range(task: MeasureTask) -> BlockMeasurement:
        return KWeightedMeter.measure_ranges(
            task.path, task.rate, [(task.start, task.end)], warmup_frames=task.warmup_frames
        )[0]

    @staticmethod
    def summarize(measurements: list[BlockMeasurement], rate: int) -> LoudnessMeasurement:
        reason = KWeightedMeter.too_quiet(measurements, rate)
        return LoudnessMeasurement(
            None if reason else KWeightedMeter.integrate(measurements),
            reason,
            KWeightedMeter.true_peak_dbtp(measurements),
        )

    def measure_file(self, path: Path, guard: ResourceGuard) -> LoudnessMeasurement:
        """Meter a float file in parallel ranges; exact to the single-pass result."""
        with sf.SoundFile(path) as audio:
            rate, frames = audio.samplerate, audio.frames
        if frames < 1:
            return LoudnessMeasurement(None, "too_short", None)
        tasks = [
            MeasureTask(str(path), rate, start, min(frames, start + self.MEASURE_RANGE_FRAMES), self.MEASURE_WARMUP_FRAMES)
            for start in range(0, frames, self.MEASURE_RANGE_FRAMES)
        ]
        return self.summarize(WorkerPool.run(self.measure_range, tasks, guard, workers=self.workers), rate)

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
                    block = audio.read(1 << 20, dtype="float32", always_2d=True)
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

    def master(
        self,
        source: Path,
        plan: CleanPlan | MasteringSettings,
        guard: ResourceGuard,
        *,
        measurement: LoudnessMeasurement | None = None,
    ) -> MasteredAudio:
        """Encode the float `source`; `measurement` skips metering when already known."""
        try:
            return self._master(source, plan, guard, measurement)
        except (OSError, RuntimeError, ValueError) as exc:
            if isinstance(exc, CleanExecutionError):
                raise
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "audio mastering failed") from exc

    def _master(
        self,
        source: Path,
        plan: CleanPlan | MasteringSettings,
        guard: ResourceGuard,
        measurement: LoudnessMeasurement | None,
    ) -> MasteredAudio:
        guard.check()
        source = source.resolve()
        if not source.is_relative_to(guard.workspace.resolve()):
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "mastering source outside workspace")
        with sf.SoundFile(source) as audio:
            rate, channels, frames = audio.samplerate, audio.channels, audio.frames
            if channels not in (1, 2) or frames < 1 or audio.subtype not in ("FLOAT", "DOUBLE"):
                raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "mastering requires float PCM")
        guard.preflight_pcm(frames, channels, copies=3, output_bytes=source.stat().st_size + 8192)
        master = source
        if rate != self.DELIVERY_RATE:
            # Piece boundaries sit on the 48 kHz frame grid, so resample once up front.
            master = guard.workspace / "master-48k.wav"
            self.runner.run(
                self._input_args(source)
                + ["-n", "-ar", str(self.DELIVERY_RATE), "-c:a", "pcm_f32le", "-threads", "1", str(master)],
                guard,
            )
            measurement = None
        with sf.SoundFile(master) as audio:
            master_frames = audio.frames
            if audio.samplerate != self.DELIVERY_RATE or audio.channels != channels:
                raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "mastering resample failed")
        measured = measurement or self.measure_file(master, guard)
        if measured.true_peak_dbtp is None and measured.unavailable_reason is None:
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "invalid loudness measurement")
        gain = 0.0
        if plan.adjust_loudness and measured.integrated_lufs is not None:
            target = -19 if channels == 1 else -16
            gain = min(6.0, target - measured.integrated_lufs)
        if measured.true_peak_dbtp is not None:
            gain = min(gain, -1.2 - measured.true_peak_dbtp)
        pieces = ParallelMp3.plan(master_frames)
        expected_delivery_frames = round(frames * self.DELIVERY_RATE / rate)
        created: list[Path] = []
        try:
            for correction in range(self.MAX_CORRECTIONS + 1):
                delivery = guard.workspace / f"delivery-{correction}.mp3"
                if delivery.exists():
                    raise CleanExecutionError(
                        ErrorCode.ARTIFACT_CONFLICT, "delivery output already exists"
                    )
                created.append(delivery)
                # One job may master several files (reconstruction segments, then the
                # final mix), so piece scratch gets a fresh directory per render.
                workspace = Path(tempfile.mkdtemp(prefix=f"mp3-{correction}-", dir=guard.workspace))
                deadline = (guard.wall_deadline or datetime.now(UTC) + timedelta(days=1)).timestamp()
                kept = WorkerPool.run(
                    ParallelMp3.encode_piece,
                    [
                        EncodeTask(str(master), str(workspace), deadline, piece, channels, master_frames, gain)
                        for piece in pieces
                    ],
                    guard,
                    workers=self.workers,
                )
                total_frames = ParallelMp3.splice([path for _, path in sorted(kept)], delivery)
                for _, path in kept:
                    Path(path).unlink(missing_ok=True)
                guard.check()
                if abs(total_frames * FRAME_SAMPLES - expected_delivery_frames) > self.DELIVERY_FRAME_TOLERANCE:
                    raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "encoded timeline mismatch")
                frame_bytes = self._fixed_frame_bytes(delivery)
                verified = WorkerPool.run(
                    ParallelMp3.verify_piece,
                    [
                        VerifyTask(str(delivery), str(workspace), deadline, piece, frame_bytes, total_frames, channels)
                        for piece in pieces
                    ],
                    guard,
                    workers=self.workers,
                )
                delivery_stats = self.summarize([item.measurement for item in verified], self.DELIVERY_RATE)
                shutil.rmtree(workspace, ignore_errors=True)
                peak = delivery_stats.true_peak_dbtp
                if peak is None or peak <= -1:
                    guard.check()
                    return MasteredAudio(
                        delivery,
                        gain,
                        rate,
                        self.DELIVERY_RATE,
                        frames,
                        channels,
                        delivery_stats,
                    )
                # Re-render from the original float intermediate, never from MP3.
                gain -= peak + 1.2
                delivery.unlink()
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "delivered true-peak gate failed")
        except BaseException:
            for path in created:
                path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _fixed_frame_bytes(delivery: Path) -> int:
        with delivery.open("rb") as stream:
            header = stream.read(4)
        length = Mp3Frames.header_length(header)
        if length is None:
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "mp3 frame stream is damaged")
        return length

