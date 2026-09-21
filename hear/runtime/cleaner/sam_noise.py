import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.runtime.cleaner.sam_features import SamFeatureFile
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


@dataclass(frozen=True)
class SamNoisePolicy:
    numpy_version: str = "1.26.4"

    @property
    def digest(self) -> str:
        descriptor = {
            "version": "sam-noise-v1",
            "numpy": self.numpy_version,
            "generator": "PCG64",
            "distribution": "standard_normal-float32",
            "seed_sequence": "[plan_seed,0x48454152,stream_id]",
            "latent_stream": 1,
            "watermark_stream": 2,
            "latent_order": "time-major-target128-residual128",
            "watermark": "integers-int64-0-inclusive-2-exclusive-[2,16]-cast-float32",
        }
        return hashlib.sha256(
            json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class SamNoise:
    def __init__(self, policy: SamNoisePolicy | None = None):
        self.policy = policy or SamNoisePolicy()
        if np.__version__ != self.policy.numpy_version:
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE, "SAM RNG NumPy version mismatch"
            )

    @staticmethod
    def _validate_seed(seed: int) -> None:
        if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
            raise ValueError("invalid SAM plan seed")

    @staticmethod
    def _generator(seed: int, stream: int):
        SamNoise._validate_seed(seed)
        try:
            return np.random.Generator(
                np.random.PCG64(np.random.SeedSequence([seed, 0x48454152, stream]))
            )
        except MemoryError:
            raise CleanExecutionError(
                ErrorCode.RESOURCE_EXHAUSTED, "SAM RNG state allocation failed"
            ) from None

    def create(
        self,
        destination: Path,
        *,
        frames: int,
        seed: int,
        guard: ResourceGuard,
        tile_frames: int = 250,
    ) -> SamFeatureFile:
        if type(frames) is not int or frames < 1:
            raise ValueError("invalid SAM noise frame count")
        if (
            type(tile_frames) is not int
            or not 1 <= tile_frames <= 65536
            or tile_frames * 256 * 4 > SamFeatureFile.MAX_TILE_BYTES
        ):
            raise ValueError("invalid SAM noise tile size")
        guard.check()
        generator = self._generator(seed, 1)
        output = SamFeatureFile(
            destination, frames=frames, batch=1, channels=256, guard=guard, create=True
        )
        try:
            for start in range(0, frames, tile_frames):
                guard.check()
                values = generator.standard_normal(
                    (min(tile_frames, frames - start), 256), dtype=np.float32
                )
                output.write(start, values.T[None])
            guard.check()
            return output
        except MemoryError:
            output.close(remove=True)
            raise CleanExecutionError(
                ErrorCode.RESOURCE_EXHAUSTED, "SAM noise allocation failed"
            ) from None
        except BaseException:
            output.close(remove=True)
            raise

    def watermark(self, seed: int) -> np.ndarray:
        try:
            return (
                self._generator(seed, 2)
                .integers(0, 2, size=(2, 16), dtype=np.int64)
                .astype(np.float32)
            )
        except MemoryError:
            raise CleanExecutionError(
                ErrorCode.RESOURCE_EXHAUSTED, "SAM watermark allocation failed"
            ) from None

    def identity(self, *, seed: int, frames: int) -> str:
        self._validate_seed(seed)
        if type(frames) is not int or frames < 1:
            raise ValueError("invalid SAM noise frame count")
        return hashlib.sha256(
            json.dumps(
                {"policy": self.policy.digest, "seed": seed, "frames": frames},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
