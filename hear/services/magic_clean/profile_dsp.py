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
        self.validate(target, guard)

    def validate(self, path: Path, guard: ResourceGuard) -> tuple[int, int]:
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
        AudioMasteringService.scan(path, guard, rate=self.RATE, channels=channels, frames=frames)
        return frames, channels

    def trim_edges(self, source: Path, target: Path, guard: ResourceGuard) -> TrimResult:
        """Trim only long near-silent edges. Preserve all internal pauses and handles."""
        frames, channels = self.validate(source, guard)
        first = None
        last = 0
        position = 0
        with sf.SoundFile(source) as audio:
            while True:
                guard.check()
                block = audio.read(480, dtype="float64", always_2d=True)
                if not len(block):
                    break
                # Any active channel preserves the block. Low thresholds protect
                # quiet consonants/breaths; these are not a speech detector.
                rms = float(np.max(np.sqrt(np.mean(block * block, axis=0))))
                peak = float(np.max(np.abs(block)))
                if rms > 10 ** (-65 / 20) or peak > 10 ** (-55 / 20):
                    if first is None:
                        first = position
                    last = position + len(block)
                position += len(block)
        handle = round(0.25 * self.RATE)
        minimum = round(0.75 * self.RATE)
        start = max(0, first - handle) if first is not None and first >= minimum else 0
        end = (
            min(frames, last + handle) if first is not None and frames - last >= minimum else frames
        )
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
