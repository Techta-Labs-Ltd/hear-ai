"""Bounded disk-backed DSP. No hard gate, source separation or speech synthesis."""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from hear.contracts.cleaning import PROFILES, MagicCleanProfile
from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.runtime.cleaner.subprocesses import CancellableProcessRunner
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode
from hear.services.magic_clean.mastering import AudioMasteringService


@dataclass(frozen=True)
class TrimResult:
    start_frame: int
    end_frame: int
    original_frames: int


class ProfileDspService:
    RATE = 48000

    def __init__(self, runner: CancellableProcessRunner):
        self.runner = runner

    @staticmethod
    def is_ready() -> bool:
        required = {
            "highpass",
            "adeclick",
            "equalizer",
            "dynaudnorm",
            "acompressor",
            "deesser",
            "atrim",
            "asetpts",
            "ebur128",
            "volume",
        }
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-filters"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
                env={"PATH": os.defpath, "LANG": "C.UTF-8"},
            )
            installed = {
                parts[1] for line in result.stdout.splitlines() if len(parts := line.split()) >= 2
            }
            return required.issubset(installed)
        except (OSError, subprocess.SubprocessError):
            return False

    @staticmethod
    def preparation_filters(options: dict) -> list[str]:
        spec = PROFILES[MagicCleanProfile(options["profile"])]
        filters = []
        if spec.highpass_hz:
            filters.append(f"highpass=f={spec.highpass_hz}:p=2")
        if options["remove_clicks"]:
            # Conservative impulse repair; not a promise to remove every mouth sound.
            filters.append("adeclick=w=40:o=75:t=3:b=2")
        return filters

    @staticmethod
    def finishing_filters(options: dict) -> list[str]:
        spec = PROFILES[MagicCleanProfile(options["profile"])]
        filters = []
        if spec.presence_db:
            filters.append(f"equalizer=f=3000:t=q:w=0.7:g={spec.presence_db}")
        if options["auto_level"]:
            # Linked-channel, bounded slow gain; do not amplify digital silence.
            filters.append("dynaudnorm=f=250:g=31:p=0.9:m=3:n=1:c=0:t=0.005")
        if spec.compression_ratio > 1:
            filters.append(
                "acompressor=threshold=0.125:ratio="
                f"{spec.compression_ratio}:attack=15:release=180:makeup=1:knee=2.8:link=maximum"
            )
            filters.append("deesser=i=0.15:m=0.25:f=0.5")
        return filters

    def render(
        self,
        source: Path,
        target: Path,
        filters: list[str],
        guard: ResourceGuard,
        *,
        decode: bool = False,
    ) -> None:
        guard.check_scratch()
        command = AudioMasteringService._input_args(source) + ["-n"]
        if filters:
            command += ["-af", ",".join(filters)]
        if decode:
            # An oversized input deliberately exceeds the validation limit, rather
            # than being accepted as a silently truncated successful recording.
            command += ["-t", str(guard.budget.max_frames / self.RATE + 0.1)]
        command += ["-ar", str(self.RATE), "-c:a", "pcm_f32le", "-threads", "1", str(target)]
        self.runner.run(command, guard)
        self.validate(target, guard, scan=not decode)

    def validate(self, path: Path, guard: ResourceGuard, *, scan: bool = True) -> tuple[int, int]:
        """Layout and length checks; `scan=False` leaves sample integrity to range readers."""
        with sf.SoundFile(path) as audio:
            frames, channels = audio.frames, audio.channels
            if audio.samplerate != self.RATE or channels not in (1, 2) or frames < 1:
                raise CleanExecutionError(
                    ErrorCode.INVALID_AUDIO, "unsupported cleaned audio layout"
                )
            if frames > guard.budget.max_frames:
                raise CleanExecutionError(
                    ErrorCode.RESOURCE_EXHAUSTED, "audio duration exceeds limit"
                )
        if scan:
            AudioMasteringService.scan(path, guard, rate=self.RATE, channels=channels, frames=frames)
        return frames, channels

    EDGE_BLOCK = 480
    EDGE_RMS = 10 ** (-65 / 20)
    EDGE_PEAK = 10 ** (-55 / 20)

    @classmethod
    def _active_blocks(cls, data: np.ndarray) -> np.ndarray:
        """Which 10 ms blocks of `data` carry signal. Any active channel keeps the block.

        Low thresholds protect quiet consonants/breaths; this is not a speech detector.
        """
        blocks = len(data) // cls.EDGE_BLOCK
        if blocks == 0:
            return np.zeros(0, dtype=bool)
        shaped = data[: blocks * cls.EDGE_BLOCK].reshape(blocks, cls.EDGE_BLOCK, -1).astype(np.float64)
        rms = np.sqrt(np.mean(shaped * shaped, axis=1)).max(axis=1)
        peak = np.abs(shaped).max(axis=(1, 2))
        return (rms > cls.EDGE_RMS) | (peak > cls.EDGE_PEAK)

    @classmethod
    def first_active_frame(cls, path: Path, guard: ResourceGuard, *, limit_frames: int) -> int | None:
        """First active block start within `limit_frames` of the file start, else None."""
        with sf.SoundFile(path) as audio:
            position = 0
            while position < min(audio.frames, limit_frames):
                guard.check()
                audio.seek(position)
                data = audio.read(cls.EDGE_BLOCK * 2000, dtype="float32", always_2d=True)
                if not len(data):
                    break
                active = cls._active_blocks(data)
                hits = np.flatnonzero(active)
                if len(hits):
                    return position + int(hits[0]) * cls.EDGE_BLOCK
                tail = len(data) % cls.EDGE_BLOCK
                if tail and (np.abs(data[-tail:]).max() > cls.EDGE_PEAK):
                    return position + len(data) - tail
                position += len(data)
        return None

    @classmethod
    def last_active_frame(cls, path: Path, guard: ResourceGuard, *, limit_frames: int) -> int | None:
        """End of the last active block within `limit_frames` of the file end, else None."""
        with sf.SoundFile(path) as audio:
            frames = audio.frames
            position = frames
            floor = max(0, frames - limit_frames)
            while position > floor:
                guard.check()
                start = max(floor, position - cls.EDGE_BLOCK * 2000)
                start -= start % cls.EDGE_BLOCK
                audio.seek(start)
                data = audio.read(position - start, dtype="float32", always_2d=True)
                active = cls._active_blocks(data)
                tail = len(data) % cls.EDGE_BLOCK
                if tail and (np.abs(data[-tail:]).max() > cls.EDGE_PEAK):
                    return position
                hits = np.flatnonzero(active)
                if len(hits):
                    return start + (int(hits[-1]) + 1) * cls.EDGE_BLOCK
                position = start
        return None

    def trim_bounds(self, source: Path, guard: ResourceGuard, frames: int) -> TrimResult:
        """Bounds that drop only long near-silent edges; internal pauses stay."""
        first = self.first_active_frame(source, guard, limit_frames=frames)
        last = self.last_active_frame(source, guard, limit_frames=frames) or 0
        handle = round(0.25 * self.RATE)
        minimum = round(0.75 * self.RATE)
        start = max(0, first - handle) if first is not None and first >= minimum else 0
        end = (
            min(frames, last + handle) if first is not None and frames - last >= minimum else frames
        )
        return TrimResult(start, end, frames)

    def trim_edges(self, source: Path, target: Path, guard: ResourceGuard) -> TrimResult:
        """Trim only long near-silent edges. Preserve all internal pauses and handles."""
        frames, channels = self.validate(source, guard)
        bounds = self.trim_bounds(source, guard, frames)
        start, end = bounds.start_frame, bounds.end_frame
        self.render(
            source,
            target,
            [f"atrim=start_sample={start}:end_sample={end}", "asetpts=PTS-STARTPTS"],
            guard,
        )
        AudioMasteringService.scan(
            target, guard, rate=self.RATE, channels=channels, frames=end - start
        )
        return TrimResult(start, end, frames)
