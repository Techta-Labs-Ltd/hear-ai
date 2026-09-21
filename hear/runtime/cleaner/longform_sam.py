"""Disk-backed global midpoint integration for the SAM long-form adapter.

This is solver infrastructure, not a certified SAM engine. Codec windows,
conditioning, pinned noise and actual SAM forward calls belong to the adapter.
No window may observe another window's update within a solver evaluation.
"""

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from hear.runtime.cleaner.mapped_residency import MappedResidency
from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


class SamVectorField(Protocol):
    def evaluate(self, state: np.ndarray, *, start_frame: int, time: float) -> np.ndarray:
        """Return float32 [latent_frames, channels] using bounded pinned conditioning."""
        ...


@dataclass(frozen=True)
class SolverPolicy:
    window_frames: int
    overlap_frames: int
    steps: int = 16

    def __post_init__(self):
        if not 2 <= self.window_frames <= 65536:
            raise ValueError("invalid latent window size")
        if not 0 < self.overlap_frames < self.window_frames:
            raise ValueError("invalid latent overlap")
        if not 1 <= self.steps <= 256:
            raise ValueError("invalid midpoint step count")

    @property
    def digest(self) -> str:
        payload = {
            "algorithm": "global-windowed-midpoint-v1",
            "window_frames": self.window_frames,
            "overlap_frames": self.overlap_frames,
            "steps": self.steps,
            "grid_origin": 0,
            "blend": "positive-linear-edge-ramp",
            "state_precision": "float32",
            "integration_interval": [0, 1],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class WindowedMidpointSolver:
    def __init__(self, policy: SolverPolicy):
        self.policy = policy

    def solve(
        self,
        noise: Path,
        destination: Path,
        *,
        frames: int,
        channels: int,
        field: SamVectorField,
        guard: ResourceGuard,
    ) -> None:
        guard.check()
        MappedResidency.require_supported()
        for path in (noise, destination):
            if path.is_symlink() or not path.resolve().is_relative_to(guard.workspace.resolve()):
                raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "latent path outside workspace")
        if destination.exists():
            raise CleanExecutionError(ErrorCode.ARTIFACT_CONFLICT, "latent output already exists")
        if not 0 < frames <= guard.budget.max_frames or not 0 < channels <= 512:
            raise CleanExecutionError(ErrorCode.RESOURCE_EXHAUSTED, "unsupported latent shape")
        size = frames * channels * 4
        if noise.stat().st_size != size:
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "latent noise size mismatch")
        occupied = sum(path.stat().st_size for path in guard.workspace.rglob("*") if path.is_file())
        if occupied + size * 3 + frames * 4 > guard.budget.scratch_bytes:
            raise CleanExecutionError(
                ErrorCode.RESOURCE_EXHAUSTED, "latent scratch reservation failed"
            )
        maps = []
        owns_destination = False
        try:
            # Exclusive destination ownership, including all partial-failure cleanup.
            with destination.open("xb") as output:
                owns_destination = True
                output.truncate(size)
            with tempfile.TemporaryDirectory(prefix="sam-solver-", dir=guard.workspace) as temp:
                state = np.memmap(
                    destination, dtype=np.float32, mode="r+", shape=(frames, channels)
                )
                maps.append(state)
                initial = np.memmap(noise, dtype=np.float32, mode="r", shape=(frames, channels))
                maps.append(initial)
                midpoint = np.memmap(
                    Path(temp) / "midpoint", dtype=np.float32, mode="w+", shape=state.shape
                )
                maps.append(midpoint)
                derivative = np.memmap(
                    Path(temp) / "derivative", dtype=np.float32, mode="w+", shape=state.shape
                )
                maps.append(derivative)
                weights = np.memmap(
                    Path(temp) / "weights", dtype=np.float32, mode="w+", shape=(frames,)
                )
                maps.append(weights)
                for start in range(0, frames, self.policy.window_frames):
                    guard.check()
                    end = min(frames, start + self.policy.window_frames)
                    if not np.isfinite(initial[start:end]).all():
                        raise CleanExecutionError(
                            ErrorCode.INVALID_AUDIO, "non-finite latent noise"
                        )
                    state[start:end] = initial[start:end]
                    self._evict(guard, state, initial)
                step = 1.0 / self.policy.steps
                for index in range(self.policy.steps):
                    self._evaluate(state, derivative, weights, field, index * step, guard)
                    self._update(state, derivative, midpoint, step / 2, guard)
                    self._evaluate(
                        midpoint, derivative, weights, field, (index + 0.5) * step, guard
                    )
                    self._update(state, derivative, state, step, guard)
                state.flush()
                guard.check()
        except BaseException:
            if owns_destination:
                destination.unlink(missing_ok=True)
            raise
        finally:
            for mapping in reversed(maps):
                mapping._mmap.close()

    def _evaluate(self, state, derivative, weights, field, time, guard):
        frames = len(state)
        window = self.policy.window_frames
        for start in range(0, frames, window):
            guard.check()
            derivative[start : start + window] = 0
            weights[start : start + window] = 0
            self._evict(guard, derivative, weights)
        stride = window - self.policy.overlap_frames
        for start in range(0, frames, stride):
            guard.check()
            end = min(frames, start + window)
            # Copy prevents a backend from modifying global solver state in place.
            prediction = field.evaluate(np.array(state[start:end]), start_frame=start, time=time)
            guard.check()
            if prediction.shape != state[start:end].shape or prediction.dtype != np.float32:
                raise CleanExecutionError(
                    ErrorCode.INVALID_AUDIO, "invalid latent derivative shape"
                )
            if not np.isfinite(prediction).all():
                raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "non-finite latent derivative")
            ramp = np.minimum(np.arange(end - start) + 1, np.arange(end - start, 0, -1))
            blend = np.minimum(ramp / self.policy.overlap_frames, 1).astype(np.float32)
            derivative[start:end] += prediction * blend[:, None]
            weights[start:end] += blend
            self._evict(guard, state, derivative, weights)
        for start in range(0, frames, window):
            guard.check()
            end = min(frames, start + window)
            if np.any(weights[start:end] <= 0):
                raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "uncovered latent frames")
            derivative[start:end] /= weights[start:end, None]
            self._evict(guard, derivative, weights)

    def _update(self, state, derivative, output, step, guard):
        for start in range(0, len(state), self.policy.window_frames):
            guard.check()
            end = min(len(state), start + self.policy.window_frames)
            result = state[start:end] + step * derivative[start:end]
            if not np.isfinite(result).all():
                raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "non-finite solver update")
            output[start:end] = result
            self._evict(guard, state, derivative, output)

    @staticmethod
    def _evict(guard, *maps):
        MappedResidency.evict(guard, *maps)
