"""Bounded integrity and conservative content-risk screening for v2.

Signal comparisons are warnings, not proof that every word is preserved.
Perceptual certification and target-speech analysis remain separate gates.
"""

from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf

from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.services.magic_clean.contracts import (
    CleanExecutionError,
    CleanPlan,
    ContentWarningInterval,
    ErrorCode,
    SourceIdentity,
    SpeechRiskEvidence,
    ValidationSummary,
)


class SpeechRiskAnalyser(Protocol):
    def evaluate(
        self, source: Path, processed: Path, expected: SourceIdentity, guard: ResourceGuard
    ) -> SpeechRiskEvidence: ...


class AudioQualityGate:
    def __init__(self, speech: SpeechRiskAnalyser | None = None):
        self.speech = speech

    def evaluate(
        self,
        source: Path,
        processed: Path,
        plan: CleanPlan,
        guard: ResourceGuard,
        *,
        expected_source: SourceIdentity | None = None,
    ) -> ValidationSummary:
        warnings = {"wanted_content_requires_review"}
        intervals = []
        truncated = False
        with sf.SoundFile(source) as original, sf.SoundFile(processed) as output:
            expected_channels = 1 if plan.profile == "voice_focus" else original.channels
            if (
                output.frames != original.frames
                or output.samplerate != original.samplerate
                or output.channels != expected_channels
                or output.subtype not in ("FLOAT", "DOUBLE")
            ):
                raise CleanExecutionError(
                    ErrorCode.INVALID_AUDIO, "engine output timeline or layout mismatch"
                )
            frames = 0
            target_detected = False
            while True:
                guard.check()
                before = original.read(32768, dtype="float64", always_2d=True)
                after = output.read(32768, dtype="float64", always_2d=True)
                if len(before) != len(after) or not np.isfinite(after).all():
                    raise CleanExecutionError(
                        ErrorCode.INVALID_AUDIO, "invalid engine output samples"
                    )
                if not len(before):
                    break
                if not np.isfinite(before).all():
                    raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "invalid source samples")
                frames += len(before)
                if np.max(np.abs(after)) > 4:
                    raise CleanExecutionError(
                        ErrorCode.INVALID_AUDIO, "unsafe engine output amplitude"
                    )
                original_rms = np.sqrt(np.mean(before * before, axis=0))
                output_rms = np.sqrt(np.mean(after * after, axis=0))
                target_detected = target_detected or bool(np.max(output_rms) > 1e-8)
                if np.max(original_rms) < 1e-8 and np.max(output_rms) > 1e-4:
                    raise CleanExecutionError(
                        ErrorCode.INVALID_AUDIO, "generated content on silent source"
                    )
                if plan.profile != "voice_focus":
                    active = original_rms > 1e-4
                    if np.any(active & (output_rms < original_rms * 0.01)):
                        raise CleanExecutionError(
                            ErrorCode.INVALID_AUDIO, "channel content disappeared"
                        )
                    if np.any(active & (output_rms < original_rms * 0.5)):
                        warnings.add("possible_wanted_content_loss")
                elif np.max(original_rms) > 1e-4 and np.max(output_rms) < 1e-8:
                    warnings.add("no_speech_target_detected")
                code = None
                ratio = 1.0
                if plan.profile != "voice_focus":
                    if np.any(active):
                        ratio = float(np.min(output_rms[active] / original_rms[active]))
                        if ratio < 0.5:
                            code = "possible_wanted_content_loss"
                elif np.max(original_rms) > 1e-4 and np.max(output_rms) < 1e-8:
                    ratio = float(np.max(output_rms) / np.max(original_rms))
                    code = "no_speech_target_detected"
                if code:
                    start = frames - len(before)
                    if (
                        intervals
                        and intervals[-1].code == code
                        and intervals[-1].end_frame == start
                    ):
                        previous = intervals.pop()
                        intervals.append(
                            ContentWarningInterval(
                                start_frame=previous.start_frame,
                                end_frame=frames,
                                code=code,
                                minimum_rms_ratio=min(previous.minimum_rms_ratio, ratio),
                            )
                        )
                    elif len(intervals) < 128:
                        intervals.append(
                            ContentWarningInterval(
                                start_frame=start,
                                end_frame=frames,
                                code=code,
                                minimum_rms_ratio=ratio,
                            )
                        )
                    else:
                        truncated = True
            if frames != original.frames:
                raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "engine decode is incomplete")
            if plan.profile == "voice_focus" and not target_detected:
                raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "no speech target detected")
        speech = None
        if self.speech is None:
            warnings.add("speech_activity_unavailable")
        else:
            if expected_source is None:
                raise CleanExecutionError(
                    ErrorCode.SOURCE_MISMATCH, "speech check needs pinned source"
                )
            try:
                speech = self.speech.evaluate(source, processed, expected_source, guard)
            except CleanExecutionError:
                raise
            except Exception:
                raise CleanExecutionError(
                    ErrorCode.PROCESS_FAILED, "configured speech risk check failed"
                ) from None
            if speech.source_loss_intervals:
                warnings.add("possible_speech_loss")
            for channel, active in enumerate(speech.source_active_frames):
                target = speech.output_active_frames[
                    channel
                    if len(speech.output_active_frames) == len(speech.source_active_frames)
                    else 0
                ]
                if active and target < active * 0.5:
                    warnings.add("possible_speech_loss")
            if speech.evidence_truncated:
                warnings.add("speech_evidence_incomplete")
        return ValidationSummary(
            hard_integrity="passed",
            wanted_content="review_required",
            warning_codes=tuple(sorted(warnings)),
            source_warning_intervals=tuple(intervals),
            warning_intervals_truncated=truncated,
            speech_activity=speech,
        )
