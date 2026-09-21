"""Conservative linked-channel spectral attenuation with explicit noise evidence.

Engineering implementation, not a listening-certified profile. Window/preset
changes require a new runtime digest. All transforms and overlap buffers are
bounded independently of recording duration.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf

from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.services.magic_clean.contracts import (
    CleanExecutionError,
    CleanPlan,
    ErrorCode,
    RuntimeIdentity,
)


@dataclass(frozen=True)
class NoiseReferenceAssessment:
    analysis_sha256: str
    speech_detected: bool
    music_detected: bool
    uncertain: bool


class NoiseReferenceAnalyser(Protocol):
    """Must inspect the selected interval, using bounded CPU-only analysis.

    A negative VAD result alone must not be returned as confident noise evidence.
    The assessment digest binds the evidence to the pinned source and interval.
    """

    @property
    def policy_sha256(self) -> str:
        """Digest of reference classification/uncertainty policy and its dependencies."""
        ...

    def assess(
        self, source: Path, plan: CleanPlan, guard: ResourceGuard
    ) -> NoiseReferenceAssessment: ...


class NoiseProfileEngine:
    def __init__(self, identity: RuntimeIdentity, analyser: NoiseReferenceAnalyser):
        if identity.engine != "noise_profile":
            raise ValueError("noise profile engine requires CPU runtime identity")
        self._identity = identity
        self._analyser = analyser
        self.validate_identity(identity, analyser)

    @staticmethod
    def describe(analyser: NoiseReferenceAnalyser) -> RuntimeIdentity:
        reference = analyser.policy_sha256
        if len(reference) != 64 or any(v not in "0123456789abcdef" for v in reference):
            raise ValueError("noise reference policy must have a SHA-256 digest")
        precision = {
            "input_scan": "float64",
            "spectrum": "complex128",
            "accumulation": "float64",
            "output": "RF64-float32",
        }
        longform = {
            "policy": "linked-channel-overlap-add-v1",
            "window": NoiseProfileSession.WINDOW,
            "hop": NoiseProfileSession.HOP,
            "window_function": "sqrt-periodic-hann",
            "gain_smoothing_bins": NoiseProfileSession.SMOOTHING_BINS,
            "gain_history_weight": NoiseProfileSession.HISTORY_WEIGHT,
            "gain_current_weight": NoiseProfileSession.CURRENT_WEIGHT,
            "channel_link": "minimum-noise-to-signal-ratio",
            "tail": "zero-pad-normalized-overlap-exact-source-frames",
        }
        runtime = {
            "implementation": "cpu-noise-profile-v2",
            "numpy": np.__version__,
            "soundfile": sf.__version__,
            "libsndfile": sf.__libsndfile_version__,
            "reference_policy_sha256": reference,
            "noise_profile": "online-mean-window-power",
            "allowed_reduction_db": [3, 6],
        }
        descriptors = [runtime, precision, longform]
        hashes = [
            hashlib.sha256(
                json.dumps(v, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            for v in descriptors
        ]
        return RuntimeIdentity(
            engine="noise_profile",
            runtime_sha256=hashes[0],
            checkpoint_sha256=None,
            precision_policy_sha256=hashes[1],
            longform_policy_sha256=hashes[2],
        )

    @classmethod
    def validate_identity(cls, identity: RuntimeIdentity, analyser: NoiseReferenceAnalyser) -> None:
        try:
            matches = identity == cls.describe(analyser)
        except (ValueError, TypeError, AttributeError):
            matches = False
        if not matches:
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE, "noise runtime provenance mismatch"
            ) from None

    @property
    def identity(self) -> RuntimeIdentity:
        return self._identity

    def open_session(self, plan: CleanPlan, guard: ResourceGuard):
        guard.check()
        if plan.runtime != self.identity or plan.profile != "music_atmosphere":
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE, "noise runtime identity mismatch"
            )
        self.validate_identity(self.identity, self._analyser)
        return NoiseProfileSession(plan, self._analyser)


class NoiseProfileSession:
    WINDOW = 2048
    HOP = 512
    SMOOTHING_BINS = 5
    HISTORY_WEIGHT = 0.8
    CURRENT_WEIGHT = 0.2

    def __init__(self, plan: CleanPlan, analyser: NoiseReferenceAnalyser):
        self.plan = plan
        self.analyser = analyser
        self.closed = False

    def close(self) -> None:
        self.closed = True

    @staticmethod
    def _finite(block: np.ndarray) -> None:
        if not np.isfinite(block).all():
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "non-finite noise-profile input")

    def _profile(self, audio: sf.SoundFile, window: np.ndarray, guard: ResourceGuard) -> np.ndarray:
        reference = self.plan.noise_reference
        assert reference is not None
        length = reference.end_frame - reference.start_frame
        if reference.end_frame > audio.frames or length < self.WINDOW:
            raise CleanExecutionError(
                ErrorCode.INVALID_AUDIO, "noise reference out of bounds or too short"
            )
        power = np.zeros((self.WINDOW // 2 + 1, audio.channels), dtype=np.float64)
        count = 0
        for offset in range(reference.start_frame, reference.end_frame - self.WINDOW + 1, self.HOP):
            guard.check()
            audio.seek(offset)
            block = audio.read(self.WINDOW, dtype="float64", always_2d=True)
            self._finite(block)
            if len(block) != self.WINDOW:
                raise CleanExecutionError(
                    ErrorCode.INVALID_AUDIO, "noise reference decode truncated"
                )
            spectrum = np.fft.rfft(block * window[:, None], axis=0)
            count += 1
            # Online mean avoids growth with reference duration.
            power += (np.abs(spectrum) ** 2 - power) / count
        if not np.any(power > 1e-20):
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "noise reference is digital silence")
        return power

    def process(
        self, source: Path, destination: Path, plan: CleanPlan, guard: ResourceGuard
    ) -> None:
        guard.check()
        if self.closed or plan != self.plan:
            raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "invalid noise session")
        NoiseProfileEngine.validate_identity(plan.runtime, self.analyser)
        for path in (source, destination):
            if path.is_symlink() or not path.resolve().is_relative_to(guard.workspace.resolve()):
                raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "audio path outside workspace")
        if destination.exists():
            raise CleanExecutionError(ErrorCode.ARTIFACT_CONFLICT, "output already exists")
        reference = plan.noise_reference
        assert reference is not None and plan.noise_reduction_db is not None
        assessment = self.analyser.assess(source, plan, guard)
        if (
            assessment.analysis_sha256 != reference.analysis_sha256
            or assessment.speech_detected
            or assessment.music_detected
            or (assessment.uncertain and not reference.confirmed_noise_only)
        ):
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "noise reference evidence rejected")
        try:
            self._process(source, destination, guard)
        except (OSError, RuntimeError, ValueError) as exc:
            destination.unlink(missing_ok=True)
            if isinstance(exc, CleanExecutionError):
                raise
            raise CleanExecutionError(
                ErrorCode.INVALID_AUDIO, "noise-profile processing failed"
            ) from exc
        except BaseException:
            # Only this invocation's newly created attempt-local output is removed.
            destination.unlink(missing_ok=True)
            raise

    def _process(self, source: Path, destination: Path, guard: ResourceGuard) -> None:
        with sf.SoundFile(source) as audio:
            if audio.channels not in (1, 2) or audio.frames < 1:
                raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "unsupported source layout")
            guard.preflight_pcm(
                audio.frames, audio.channels, copies=2, output_bytes=source.stat().st_size + 4096
            )
            window = np.sqrt(np.hanning(self.WINDOW + 1)[:-1])
            noise = self._profile(audio, window, guard)
            floor = 10 ** (-self.plan.noise_reduction_db / 20)
            overlap = np.zeros((self.WINDOW, audio.channels), dtype=np.float64)
            weights = np.zeros(self.WINDOW, dtype=np.float64)
            previous_gain = np.ones(self.WINDOW // 2 + 1)
            written = 0
            with sf.SoundFile(
                destination,
                "w",
                samplerate=audio.samplerate,
                channels=audio.channels,
                format="RF64",
                subtype="FLOAT",
            ) as output:
                for start in range(-self.WINDOW + self.HOP, audio.frames, self.HOP):
                    guard.check()
                    block = np.zeros((self.WINDOW, audio.channels), dtype=np.float64)
                    left = max(0, start)
                    right = min(audio.frames, start + self.WINDOW)
                    if right > left:
                        audio.seek(left)
                        data = audio.read(right - left, dtype="float64", always_2d=True)
                        self._finite(data)
                        if len(data) != right - left:
                            raise CleanExecutionError(
                                ErrorCode.INVALID_AUDIO, "source decode truncated"
                            )
                        block[left - start : right - start] = data
                    spectrum = np.fft.rfft(block * window[:, None], axis=0)
                    power = np.abs(spectrum) ** 2
                    # A bin protected by either channel is protected in both.
                    ratio = np.min(noise / np.maximum(power, 1e-20), axis=1)
                    gain = np.sqrt(np.maximum(0, 1 - np.minimum(ratio, 1)))
                    gain = np.maximum(floor, gain)
                    half = self.SMOOTHING_BINS // 2
                    gain = np.convolve(
                        np.pad(gain, (half, half), mode="edge"),
                        np.ones(self.SMOOTHING_BINS) / self.SMOOTHING_BINS,
                        "valid",
                    )
                    # Immediate protection of transients; slower attenuation onset.
                    gain = np.maximum(
                        gain, self.HISTORY_WEIGHT * previous_gain + self.CURRENT_WEIGHT * gain
                    )
                    previous_gain = gain
                    filtered = np.fft.irfft(spectrum * gain[:, None], n=self.WINDOW, axis=0)
                    overlap += filtered * window[:, None]
                    weights += window * window
                    begin = max(0, -start)
                    end = min(self.HOP, audio.frames - start)
                    if end > begin:
                        chunk = overlap[begin:end] / weights[begin:end, None]
                        self._finite(chunk)
                        output.write(chunk)
                        written += len(chunk)
                    overlap[: -self.HOP] = overlap[self.HOP :]
                    overlap[-self.HOP :] = 0
                    weights[: -self.HOP] = weights[self.HOP :]
                    weights[-self.HOP :] = 0
                if written != audio.frames:
                    raise CleanExecutionError(
                        ErrorCode.INVALID_AUDIO, "noise processing changed timeline"
                    )
        guard.check()
