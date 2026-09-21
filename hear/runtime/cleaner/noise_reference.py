import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.runtime.cleaner.speech_activity import CpuSpeechActivity
from hear.services.magic_clean.contracts import (
    CleanExecutionError,
    CleanPlan,
    ErrorCode,
    NoiseReferenceSelection,
    SourceIdentity,
)
from hear.services.magic_clean.engines.noise_profile import NoiseReferenceAssessment


@dataclass(frozen=True)
class NoiseReferenceReview:
    assessment: NoiseReferenceAssessment
    source_sha256: str
    revision_id: str
    start_frame: int
    end_frame: int
    analysis_policy_sha256: str
    speech_active_frames: tuple[int, ...]
    warning_codes: tuple[str, ...]


class SpeechAwareNoiseReferenceAnalyser:
    POLICY = "selected-noise-reference-v1-speech-veto-music-unknown-confirmation-required"

    def __init__(self, speech: CpuSpeechActivity):
        self.speech = speech

    @property
    def policy_sha256(self) -> str:
        descriptor = {"policy": self.POLICY, "speech_policy_sha256": self.speech.policy.digest}
        return hashlib.sha256(
            json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @staticmethod
    def _hash(path: Path, guard: ResourceGuard) -> tuple[str, int]:
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                guard.check()
                size += len(chunk)
                if size > guard.budget.max_input_bytes:
                    raise CleanExecutionError(ErrorCode.RESOURCE_EXHAUSTED, "noise input too large")
                digest.update(chunk)
        return digest.hexdigest(), size

    def assess(
        self, source: Path, plan: CleanPlan, guard: ResourceGuard
    ) -> NoiseReferenceAssessment:
        reference = plan.noise_reference
        if reference is None:
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "selected noise reference required")
        return self.review(source, reference, guard).assessment

    def review(
        self, source: Path, reference: NoiseReferenceSelection, guard: ResourceGuard
    ) -> NoiseReferenceReview:
        guard.check()
        if (
            source.is_symlink()
            or not source.is_file()
            or not source.resolve().is_relative_to(guard.workspace.resolve())
        ):
            raise CleanExecutionError(ErrorCode.SOURCE_MISMATCH, "noise source outside workspace")
        try:
            return self._review(source, reference, guard)
        except CleanExecutionError:
            raise
        except (OSError, RuntimeError, ValueError):
            raise CleanExecutionError(
                ErrorCode.INVALID_AUDIO, "noise reference analysis failed"
            ) from None

    def _review(
        self, source: Path, reference: NoiseReferenceSelection, guard: ResourceGuard
    ) -> NoiseReferenceReview:
        source_hash, _ = self._hash(source, guard)
        length = reference.end_frame - reference.start_frame
        with tempfile.TemporaryDirectory(prefix="noise-reference-", dir=guard.workspace) as temp:
            crop = Path(temp) / "reference.wav"
            with sf.SoundFile(source) as audio:
                if (
                    audio.channels not in (1, 2)
                    or not 0 < audio.frames <= guard.budget.max_frames
                    or not 8000 <= audio.samplerate <= 96000
                    or reference.end_frame > audio.frames
                    or length < 2048
                ):
                    raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "invalid reference interval")
                rate, channels = audio.samplerate, audio.channels
                guard.preflight_pcm(
                    length, channels, copies=2, output_bytes=source.stat().st_size + 4096
                )
                audio.seek(reference.start_frame)
                remaining = length
                peak = 0.0
                with sf.SoundFile(
                    crop, "w", samplerate=rate, channels=channels, format="RF64", subtype="FLOAT"
                ) as output:
                    while remaining:
                        guard.check()
                        block = audio.read(min(32768, remaining), dtype="float32", always_2d=True)
                        if not len(block) or not np.isfinite(block).all():
                            raise CleanExecutionError(
                                ErrorCode.INVALID_AUDIO, "invalid reference PCM"
                            )
                        peak = max(peak, float(np.max(np.abs(block))))
                        output.write(block)
                        remaining -= len(block)
                if peak <= 1e-10:
                    raise CleanExecutionError(
                        ErrorCode.INVALID_AUDIO, "reference is digital silence"
                    )
            crop_hash, crop_size = self._hash(crop, guard)
            # Private attempt-local analysis identity, not a backend media object.
            identity = SourceIdentity(
                revision_id="selected-reference",
                media_id="selected-reference",
                object_key="attempt-local/reference.wav",
                object_version="local",
                sha256=crop_hash,
                size_bytes=crop_size,
                sample_rate=rate,
                channels=channels,
                frames=length,
            )
            activity = self.speech.scan(crop, identity, guard)
            guard.check()
            if (
                activity.source_sha256 != crop_hash
                or (activity.sample_rate, activity.frames, activity.channels)
                != (rate, length, channels)
                or len(activity.active_frames) != channels
                or any(not 0 <= count <= length for count in activity.active_frames)
            ):
                raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "reference evidence mismatch")
        detected = any(v > 0 for v in activity.active_frames)
        warnings = ["noise_reference_requires_confirmation", "music_analysis_unavailable"]
        if detected:
            warnings.append("speech_in_noise_reference")
        descriptor = {
            "policy": self.POLICY,
            "reference_policy_sha256": self.policy_sha256,
            "analysis_policy_sha256": activity.policy_sha256,
            "source_sha256": source_hash,
            "revision_id": reference.revision_id,
            "start_frame": reference.start_frame,
            "end_frame": reference.end_frame,
            "sample_rate": rate,
            "channels": channels,
            "speech_active_frames": activity.active_frames,
            "music_assessment": "unknown",
            "uncertain": True,
        }
        digest = hashlib.sha256(
            json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return NoiseReferenceReview(
            NoiseReferenceAssessment(digest, detected, False, True),
            source_hash,
            reference.revision_id,
            reference.start_frame,
            reference.end_frame,
            activity.policy_sha256,
            activity.active_frames,
            tuple(warnings),
        )
