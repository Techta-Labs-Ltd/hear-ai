"""Bounded inspection of an already downloaded, pinned source.

Download authorization is the transport/storage adapter's responsibility.
This inspector never follows remote URLs or loads a complete waveform.
"""

import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.runtime.cleaner.subprocesses import CancellableProcessRunner
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode, SourceIdentity


@dataclass(frozen=True)
class AudioInspection:
    frames: int
    sample_rate: int
    channels: int
    peaks: tuple[float, ...]
    rms: tuple[float, ...]
    channel_correlation: float | None
    # Diagnostic evidence only: correlation is not a proof of wanted-speech retention.


class SourceInspector:
    BLOCK_FRAMES = 32768
    FORMATS = frozenset({"WAV", "WAVEX", "RF64", "FLAC", "MPEG", "OGG"})

    @staticmethod
    def inspect(path: Path, expected: SourceIdentity, guard: ResourceGuard) -> AudioInspection:
        """Supervise native decoding outside the model-owning worker process.

        The child receives no inherited environment credentials. Its bounded
        stderr is a private structured response, never an operational log.
        This is process/deadline isolation, not an OS security sandbox.
        """
        guard.check()
        response = CancellableProcessRunner(16384).run(
            [
                sys.executable,
                "-m",
                "hear.runtime.cleaner.inspection_worker",
                str(path.absolute()),
                expected.model_dump_json(),
                str(guard.workspace.absolute()),
                str(guard.deadline),
                str(guard.budget.scratch_bytes),
                str(guard.budget.max_input_bytes),
                str(guard.budget.max_frames),
            ],
            guard,
            env={
                "PATH": os.defpath,
                "PYTHONPATH": str(Path(__file__).resolve().parents[3]),
                "LANG": "C.UTF-8",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
        )
        try:
            payload = json.loads(response)
            if set(payload) == {"error"}:
                code = ErrorCode(payload["error"])
                raise CleanExecutionError(code, "source inspection failed")
            result = AudioInspection(**payload)
            if (
                result.frames != expected.frames
                or result.sample_rate != expected.sample_rate
                or result.channels != expected.channels
                or len(result.peaks) != expected.channels
                or len(result.rms) != expected.channels
                or any(not math.isfinite(v) or v < 0 for v in (*result.peaks, *result.rms))
                or (
                    result.channel_correlation is not None
                    and not -1 <= result.channel_correlation <= 1
                )
            ):
                raise ValueError("invalid inspection response")
            return AudioInspection(
                result.frames,
                result.sample_rate,
                result.channels,
                tuple(result.peaks),
                tuple(result.rms),
                result.channel_correlation,
            )
        except CleanExecutionError:
            raise
        except (TypeError, ValueError, KeyError):
            raise CleanExecutionError(
                ErrorCode.INVALID_AUDIO, "invalid source inspection response"
            ) from None

    @staticmethod
    def _inspect_local(
        path: Path, expected: SourceIdentity, guard: ResourceGuard
    ) -> AudioInspection:
        """Child-only bounded scan; callers use the supervised public method."""
        guard.check()
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(guard.workspace.resolve())
        ):
            raise CleanExecutionError(ErrorCode.SOURCE_MISMATCH, "source outside attempt workspace")
        if expected.size_bytes > guard.budget.max_input_bytes:
            raise CleanExecutionError(ErrorCode.RESOURCE_EXHAUSTED, "source exceeds input budget")
        guard.preflight_pcm(
            expected.frames, expected.channels, copies=1, output_bytes=expected.size_bytes
        )
        with path.open("rb") as source:
            digest = hashlib.sha256()
            size = 0
            while chunk := source.read(1024 * 1024):
                guard.check()
                size += len(chunk)
                if size > expected.size_bytes:
                    raise CleanExecutionError(ErrorCode.SOURCE_MISMATCH, "source size mismatch")
                digest.update(chunk)
            if size != expected.size_bytes or digest.hexdigest() != expected.sha256:
                raise CleanExecutionError(ErrorCode.SOURCE_MISMATCH, "source checksum mismatch")
            source.seek(0)
            try:
                with sf.SoundFile(source, mode="r") as audio:
                    if (
                        audio.format not in SourceInspector.FORMATS
                        or audio.channels != expected.channels
                        or audio.samplerate != expected.sample_rate
                        or audio.frames != expected.frames
                    ):
                        raise CleanExecutionError(
                            ErrorCode.INVALID_AUDIO, "source metadata mismatch"
                        )
                    peaks = np.zeros(audio.channels, dtype=np.float64)
                    squares = np.zeros(audio.channels, dtype=np.float64)
                    sums = np.zeros(audio.channels, dtype=np.float64)
                    cross = 0.0
                    frames = 0
                    while True:
                        guard.check()
                        block = audio.read(
                            SourceInspector.BLOCK_FRAMES, dtype="float64", always_2d=True
                        )
                        if not len(block):
                            break
                        frames += len(block)
                        if frames > expected.frames or not np.isfinite(block).all():
                            raise CleanExecutionError(
                                ErrorCode.INVALID_AUDIO, "invalid decoded PCM"
                            )
                        peaks = np.maximum(peaks, np.abs(block).max(axis=0))
                        squares += np.sum(block * block, axis=0)
                        sums += np.sum(block, axis=0)
                        if audio.channels == 2:
                            cross += float(np.sum(block[:, 0] * block[:, 1]))
                    if frames != expected.frames:
                        raise CleanExecutionError(
                            ErrorCode.INVALID_AUDIO, "decoded source is incomplete"
                        )
                    correlation = None
                    if audio.channels == 2:
                        variance = np.maximum(0, squares - sums * sums / frames)
                        denominator = float(np.sqrt(variance[0] * variance[1]))
                        if denominator > 0:
                            correlation = float(
                                np.clip((cross - sums[0] * sums[1] / frames) / denominator, -1, 1)
                            )
                    result = AudioInspection(
                        frames,
                        audio.samplerate,
                        audio.channels,
                        tuple(float(v) for v in peaks),
                        tuple(float(v) for v in np.sqrt(squares / frames)),
                        correlation,
                    )
            except (RuntimeError, ValueError) as exc:
                if isinstance(exc, CleanExecutionError):
                    raise
                raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "source decode failed") from exc
        guard.check()
        return result
