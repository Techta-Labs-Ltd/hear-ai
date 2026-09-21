from pathlib import Path
from typing import Literal

import soundfile as sf

from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.runtime.cleaner.sam_features import SamFeatureFile
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


class SamPCM:
    RATE = 48000

    def __init__(self, *, tile_frames: int = 48000):
        if type(tile_frames) is not int or not 1 <= tile_frames <= 65536:
            raise ValueError("invalid SAM PCM tile size")
        self.tile_frames = tile_frames

    @staticmethod
    def _path(path: Path, guard: ResourceGuard) -> None:
        guard.check()
        if path.is_symlink() or not path.resolve().is_relative_to(guard.workspace.resolve()):
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "SAM PCM path outside workspace")

    def read(self, source: Path, destination: Path, guard: ResourceGuard) -> SamFeatureFile:
        """Import prepared mono 48 kHz PCM; never implicitly downmix or resample."""
        self._path(source, guard)
        self._path(destination, guard)
        result = None
        try:
            if source.stat().st_size > guard.budget.max_input_bytes:
                raise CleanExecutionError(ErrorCode.RESOURCE_EXHAUSTED, "SAM PCM input too large")
            with sf.SoundFile(source) as audio:
                if (
                    audio.samplerate != self.RATE
                    or audio.channels != 1
                    or audio.format not in ("WAV", "WAVEX", "RF64")
                    or audio.subtype not in ("PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE")
                    or audio.frames < 1
                ):
                    raise CleanExecutionError(
                        ErrorCode.INVALID_AUDIO, "SAM requires prepared 48 kHz mono PCM"
                    )
                result = SamFeatureFile(
                    destination, frames=audio.frames, batch=1, channels=1, guard=guard, create=True
                )
                for start in range(0, audio.frames, self.tile_frames):
                    guard.check()
                    count = min(self.tile_frames, audio.frames - start)
                    samples = audio.read(count, dtype="float32", always_2d=True)
                    if samples.shape != (count, 1):
                        raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "short SAM PCM read")
                    result.write(start, samples.T[None])
                guard.check()
            return result
        except BaseException as error:
            if result is not None:
                result.close(remove=True)
            if isinstance(error, MemoryError):
                raise CleanExecutionError(
                    ErrorCode.RESOURCE_EXHAUSTED, "SAM PCM allocation failed"
                ) from None
            if isinstance(error, (OSError, sf.LibsndfileError)):
                raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "SAM PCM read failed") from None
            raise

    def write(
        self,
        source: SamFeatureFile,
        destination: Path,
        *,
        stream: Literal["target", "residual"],
    ) -> None:
        """Export one explicitly selected waveform, not a target/residual stereo mix."""
        guard = source.guard
        self._path(destination, guard)
        if (
            stream not in ("target", "residual")
            or source.batch != 2
            or source.channels != 1
            or not source.complete
        ):
            raise ValueError("SAM PCM export requires complete paired waveforms and a stream")
        occupied = sum(p.stat().st_size for p in guard.workspace.rglob("*") if p.is_file())
        if occupied + source.frames * 4 + 4096 > guard.budget.scratch_bytes:
            raise CleanExecutionError(ErrorCode.RESOURCE_EXHAUSTED, "SAM PCM scratch exhausted")
        owned = False
        try:
            with destination.open("xb") as raw:
                owned = True
                with sf.SoundFile(
                    raw, "w", samplerate=self.RATE, channels=1, format="RF64", subtype="FLOAT"
                ) as audio:
                    index = 0 if stream == "target" else 1
                    for start in range(0, source.frames, self.tile_frames):
                        values = source.read(start, min(source.frames, start + self.tile_frames))
                        guard.check()
                        audio.write(values[index].T)
                        guard.check()
            guard.check()
        except BaseException as error:
            if owned:
                destination.unlink(missing_ok=True)
            if isinstance(error, FileExistsError):
                raise CleanExecutionError(
                    ErrorCode.ARTIFACT_CONFLICT, "SAM PCM output exists"
                ) from None
            if isinstance(error, (MemoryError, OSError, sf.LibsndfileError)):
                raise CleanExecutionError(
                    ErrorCode.RESOURCE_EXHAUSTED, "SAM PCM write failed"
                ) from None
            raise
