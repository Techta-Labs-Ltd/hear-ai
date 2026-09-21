import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.runtime.cleaner.subprocesses import CancellableProcessRunner
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


class AudioResampler:
    POLICY = "swr-kaiser64-beta9-cutoff0.97-phase10-exact-linear-f32-pad4096-ceil-v1"
    OPTIONS = (
        "resampler=swr:filter_size=64:phase_shift=10:exact_rational=1:"
        "linear_interp=1:filter_type=kaiser:kaiser_beta=9:cutoff=0.97:"
        "async=0:dither_method=0:tsf=fltp"
    )

    def __init__(self, runner: CancellableProcessRunner):
        self.runner = runner

    @staticmethod
    def frame_count(frames: int, source_rate: int, target_rate: int) -> int:
        return (frames * target_rate + source_rate - 1) // source_rate

    def convert(
        self,
        source: Path,
        destination: Path,
        target_rate: int,
        guard: ResourceGuard,
        *,
        exact_frames: int | None = None,
    ) -> None:
        """Validate a private output, then publish it with create-only ownership."""
        guard.check()
        if destination.is_symlink() or not destination.resolve().is_relative_to(
            guard.workspace.resolve()
        ):
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "resample path outside workspace")
        if destination.exists():
            raise CleanExecutionError(ErrorCode.ARTIFACT_CONFLICT, "resample output already exists")
        published = None
        try:
            with tempfile.TemporaryDirectory(prefix="resample-", dir=guard.workspace) as directory:
                staged = Path(directory) / "output.wav"
                self._convert_owned(source, staged, target_rate, guard, exact_frames=exact_frames)
                guard.check()
                stat = staged.stat()
                os.link(staged, destination)
                published = (stat.st_dev, stat.st_ino)
                staged.unlink()
            guard.check()
        except BaseException as exc:
            if published is not None:
                try:
                    stat = destination.lstat()
                except FileNotFoundError:
                    pass
                else:
                    if (stat.st_dev, stat.st_ino) == published:
                        destination.unlink()
            if isinstance(exc, FileExistsError):
                raise CleanExecutionError(
                    ErrorCode.ARTIFACT_CONFLICT, "resample output already exists"
                ) from None
            if isinstance(exc, OSError):
                raise CleanExecutionError(
                    ErrorCode.PROCESS_FAILED, "resample publication failed"
                ) from None
            raise

    def _convert_owned(
        self,
        source: Path,
        destination: Path,
        target_rate: int,
        guard: ResourceGuard,
        *,
        exact_frames: int | None = None,
    ) -> None:
        guard.check()
        for path in (source, destination):
            if path.is_symlink() or not path.resolve().is_relative_to(guard.workspace.resolve()):
                raise CleanExecutionError(
                    ErrorCode.INVALID_AUDIO, "resample path outside workspace"
                )
        if destination.exists():
            raise CleanExecutionError(ErrorCode.ARTIFACT_CONFLICT, "resample output already exists")
        try:
            with sf.SoundFile(source) as audio:
                source_rate, channels, frames = audio.samplerate, audio.channels, audio.frames
                if (
                    not 8000 <= source_rate <= 96000
                    or not 8000 <= target_rate <= 96000
                    or channels not in (1, 2)
                    or frames < 1
                ):
                    raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "unsupported resample input")
                expected = self.frame_count(frames, source_rate, target_rate)
                if exact_frames is not None:
                    # Only permit the rounding correction from a source->48k->source trip.
                    tolerance = self.frame_count(1, source_rate, target_rate)
                    if not 0 < exact_frames <= expected or expected - exact_frames > tolerance:
                        raise CleanExecutionError(
                            ErrorCode.INVALID_AUDIO, "invalid resample timing"
                        )
                    expected = exact_frames
                guard.preflight_pcm(
                    expected, channels, copies=1, output_bytes=source.stat().st_size + 4096
                )
                while True:
                    guard.check()
                    data = audio.read(32768, dtype="float32", always_2d=True)
                    if not len(data):
                        break
                    if not np.isfinite(data).all():
                        raise CleanExecutionError(
                            ErrorCode.INVALID_AUDIO, "non-finite resample input"
                        )
            filters = (
                f"apad=pad_len=4096,aresample={target_rate}:{self.OPTIONS},"
                f"atrim=end_sample={expected},asetpts=N/SR/TB"
            )
            self.runner.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-nostdin",
                    "-v",
                    "error",
                    "-n",
                    "-threads",
                    "1",
                    "-filter_threads",
                    "1",
                    "-protocol_whitelist",
                    "file,pipe",
                    "-i",
                    str(source.resolve()),
                    "-map",
                    "0:a:0",
                    "-vn",
                    "-sn",
                    "-dn",
                    "-map_metadata",
                    "-1",
                    "-af",
                    filters,
                    "-c:a",
                    "pcm_f32le",
                    "-rf64",
                    "auto",
                    "-f",
                    "wav",
                    str(destination.resolve()),
                ],
                guard,
            )
            with sf.SoundFile(destination) as output:
                if (output.samplerate, output.channels, output.frames, output.subtype) != (
                    target_rate,
                    channels,
                    expected,
                    "FLOAT",
                ):
                    raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "resample output mismatch")
                while True:
                    guard.check()
                    data = output.read(32768, dtype="float32", always_2d=True)
                    if not len(data):
                        break
                    if not np.isfinite(data).all():
                        raise CleanExecutionError(
                            ErrorCode.INVALID_AUDIO, "non-finite resample output"
                        )
        except BaseException as exc:
            destination.unlink(missing_ok=True)
            if isinstance(exc, (OSError, ValueError, RuntimeError)) and not isinstance(
                exc, CleanExecutionError
            ):
                raise CleanExecutionError(ErrorCode.PROCESS_FAILED, "resampling failed") from exc
            raise
