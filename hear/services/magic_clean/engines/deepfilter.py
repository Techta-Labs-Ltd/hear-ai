"""Bounded contextual DeepFilterNet3 file adapter.

The backend must use the pinned DF3 model with delay-compensated inference; the
attenuation limit and post-filter are per-plan inference options, not model variants. Context length is part of the certified longform policy, not an
assertion of equivalence to unbounded continuous inference.
"""

import hashlib
import json
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf

from hear.runtime.cleaner.path_cleanup import AttemptPathCleanup, OwnedFileIdentity
from hear.runtime.cleaner.resampling import AudioResampler
from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.runtime.cleaner.subprocesses import CancellableProcessRunner
from hear.services.magic_clean.contracts import (
    CleanExecutionError,
    CleanPlan,
    ErrorCode,
    RuntimeIdentity,
)


class DeepFilterBackend(Protocol):
    """One attempt's model/state; each call resets state with full context.

    Input/output are channel-first float32 at exactly 48 kHz. Implementation
    must compensate analysis/synthesis delay and return exactly input length.
    No implicit device switch, extra model or attenuation default is permitted.
    """

    def enhance(
        self, samples: np.ndarray, attenuation_limit_db: int, post_filter: bool
    ) -> np.ndarray: ...

    def close(self) -> None: ...


class DeepFilterBackendFactory(Protocol):
    def validate_identity(self, identity: RuntimeIdentity) -> None: ...

    def open(self, guard: ResourceGuard) -> DeepFilterBackend: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class ContextualPolicy:
    block_frames: int
    context_frames: int

    def __post_init__(self):
        if not 48000 <= self.block_frames <= 480000 or not 4800 <= self.context_frames <= 240000:
            raise ValueError("unsupported contextual block bounds")

    @property
    def digest(self) -> str:
        payload = {
            "algorithm": "df3-context-crop-v1",
            "sample_rate": 48000,
            "block_frames": self.block_frames,
            "context_frames": self.context_frames,
            "resampling": AudioResampler.POLICY,
            "subprocess_runner": CancellableProcessRunner.POLICY,
            "subprocess_shutdown_seconds": (
                CancellableProcessRunner.DEFAULT_SHUTDOWN_TIMEOUT_SECONDS
            ),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class DeepFilterEngine:
    def __init__(
        self, identity: RuntimeIdentity, factory: DeepFilterBackendFactory, policy: ContextualPolicy
    ):
        if identity.engine != "deepfilternet3" or identity.longform_policy_sha256 != policy.digest:
            raise ValueError("DeepFilter policy/runtime mismatch")
        factory.validate_identity(identity)
        self._identity = identity
        self.factory = factory
        self.policy = policy
        self._lease = threading.Lock()
        self._closed = False

    @property
    def identity(self) -> RuntimeIdentity:
        return self._identity

    def open_session(self, plan: CleanPlan, guard: ResourceGuard):
        guard.check()
        if self._closed:
            raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "DeepFilter engine is closed")
        if plan.runtime != self.identity or plan.profile != "natural":
            raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "DeepFilter runtime mismatch")
        if not self._lease.acquire(blocking=False):
            raise CleanExecutionError(
                ErrorCode.RESOURCE_EXHAUSTED, "DeepFilter session is occupied"
            )
        try:
            self.factory.validate_identity(self.identity)
            backend = self.factory.open(guard)
        except BaseException:
            self._lease.release()
            raise
        return DeepFilterSession(plan, backend, self.policy, self._lease)

    def close(self) -> None:
        if self._closed:
            return
        if not self._lease.acquire(blocking=False):
            raise CleanExecutionError(
                ErrorCode.RESOURCE_EXHAUSTED,
                "DeepFilter engine still has an active session",
                worker_restart_required=True,
            )
        try:
            self.factory.close()
            self._closed = True
        finally:
            self._lease.release()


class DeepFilterSession:
    MODEL_RATE = 48000

    def __init__(
        self,
        plan: CleanPlan,
        backend: DeepFilterBackend,
        policy: ContextualPolicy,
        lease: threading.Lock,
    ):
        self.plan = plan
        self.backend = backend
        self.policy = policy
        self.lease = lease
        self.closed = False

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            try:
                self.backend.close()
            finally:
                self.lease.release()

    def process(
        self, source: Path, destination: Path, plan: CleanPlan, guard: ResourceGuard
    ) -> None:
        guard.check()
        if destination.is_symlink() or not destination.resolve().is_relative_to(
            guard.workspace.resolve()
        ):
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "audio path outside workspace")
        if destination.exists():
            raise CleanExecutionError(
                ErrorCode.ARTIFACT_CONFLICT, "DeepFilter output already exists"
            )
        published: OwnedFileIdentity | None = None
        try:
            with tempfile.TemporaryDirectory(
                prefix="df3-output-", dir=guard.workspace
            ) as directory:
                staged = Path(directory) / "output.wav"
                self._process_attempt(source, staged, plan, guard)
                guard.check()
                published = AttemptPathCleanup.identity(staged)
                os.link(staged, destination)
                staged.unlink()
            guard.check()
        except BaseException as exc:
            if published is not None:
                AttemptPathCleanup.remove_if_owned(destination, published, exc)
            if isinstance(exc, FileExistsError):
                raise CleanExecutionError(
                    ErrorCode.ARTIFACT_CONFLICT, "DeepFilter output already exists"
                ) from None
            if isinstance(exc, OSError):
                raise CleanExecutionError(
                    ErrorCode.PROCESS_FAILED, "DeepFilter publication failed"
                ) from None
            raise

    def _process_attempt(
        self, source: Path, destination: Path, plan: CleanPlan, guard: ResourceGuard
    ) -> None:
        guard.check()
        if self.closed or plan != self.plan:
            raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "invalid DeepFilter session")
        for path in (source, destination):
            if path.is_symlink() or not path.resolve().is_relative_to(guard.workspace.resolve()):
                raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "audio path outside workspace")
        if destination.exists():
            raise CleanExecutionError(
                ErrorCode.ARTIFACT_CONFLICT, "DeepFilter output already exists"
            )
        try:
            with sf.SoundFile(source) as audio:
                rate, frames, channels = audio.samplerate, audio.frames, audio.channels
            if rate == self.MODEL_RATE:
                self._process(source, destination, guard)
            else:
                model_frames = AudioResampler.frame_count(frames, rate, self.MODEL_RATE)
                guard.preflight_pcm(
                    model_frames,
                    channels,
                    copies=2,
                    output_bytes=source.stat().st_size + frames * channels * 4 + 16384,
                )
                resampler = AudioResampler(CancellableProcessRunner())
                with tempfile.TemporaryDirectory(
                    prefix="df3-resample-", dir=guard.workspace
                ) as temp:
                    prepared = Path(temp) / "prepared.wav"
                    enhanced = Path(temp) / "enhanced.wav"
                    resampler.convert(source, prepared, self.MODEL_RATE, guard)
                    self._process(prepared, enhanced, guard)
                    resampler.convert(enhanced, destination, rate, guard, exact_frames=frames)
        except (OSError, RuntimeError, ValueError) as exc:
            AttemptPathCleanup.remove_private(destination, exc)
            if isinstance(exc, CleanExecutionError):
                raise
            raise CleanExecutionError(
                ErrorCode.PROCESS_FAILED, "DeepFilter processing failed"
            ) from exc
        except BaseException as exc:
            AttemptPathCleanup.remove_private(destination, exc)
            raise

    def _process(self, source: Path, destination: Path, guard: ResourceGuard) -> None:
        with sf.SoundFile(source) as audio:
            if audio.samplerate != self.MODEL_RATE or audio.channels not in (1, 2):
                raise CleanExecutionError(
                    ErrorCode.INVALID_AUDIO, "DeepFilter requires prepared 48 kHz mono/stereo"
                )
            guard.preflight_pcm(
                audio.frames, audio.channels, copies=2, output_bytes=source.stat().st_size + 4096
            )
            with sf.SoundFile(
                destination,
                "w",
                samplerate=self.MODEL_RATE,
                channels=audio.channels,
                format="RF64",
                subtype="FLOAT",
            ) as output:
                for group in self._block_groups(audio.frames):
                    guard.check()
                    # Every block is cropped from its own context window, so blocks of
                    # equal length can share one forward pass as extra channels: the
                    # model treats channels independently and the result is identical.
                    stacked = []
                    for _start, _end, left, right in group:
                        audio.seek(left)
                        samples = audio.read(right - left, dtype="float32", always_2d=True).T
                        if (
                            samples.shape != (audio.channels, right - left)
                            or not np.isfinite(samples).all()
                        ):
                            raise CleanExecutionError(
                                ErrorCode.INVALID_AUDIO, "invalid DeepFilter input block"
                            )
                        stacked.append(samples)
                    batch = np.ascontiguousarray(np.concatenate(stacked, axis=0))
                    enhanced = self.backend.enhance(
                        batch, self.attenuation(self.plan), self.plan.post_filter
                    )
                    guard.check()
                    if (
                        enhanced.shape != batch.shape
                        or enhanced.dtype != np.float32
                        or not np.isfinite(enhanced).all()
                    ):
                        raise CleanExecutionError(
                            ErrorCode.INVALID_AUDIO, "DeepFilter output shape or samples invalid"
                        )
                    for index, (start, end, left, _right) in enumerate(group):
                        rows = slice(index * audio.channels, (index + 1) * audio.channels)
                        output.write(enhanced[rows, start - left : end - left].T)
        guard.check()

    BATCH_BLOCKS = 8

    def _block_groups(self, frames: int) -> list[list[tuple[int, int, int, int]]]:
        """Blocks with their context windows, grouped by equal length for batching."""
        blocks = []
        for start in range(0, frames, self.policy.block_frames):
            end = min(frames, start + self.policy.block_frames)
            left = max(0, start - self.policy.context_frames)
            right = min(frames, end + self.policy.context_frames)
            blocks.append((start, end, left, right))
        groups: list[list[tuple[int, int, int, int]]] = []
        for block in blocks:
            length = block[3] - block[2]
            if (
                groups
                and len(groups[-1]) < self.BATCH_BLOCKS
                and groups[-1][-1][3] - groups[-1][-1][2] == length
            ):
                groups[-1].append(block)
            else:
                groups.append([block])
        return groups

    @staticmethod
    def attenuation(plan: CleanPlan) -> int:
        if plan.attenuation_limit_db is None:
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "missing attenuation limit")
        return plan.attenuation_limit_db
