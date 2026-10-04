from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

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

BLOCK_FRAMES = 32768


class SpeechRiskAnalyser(Protocol):
    def evaluate(
        self, source: Path, processed: Path, expected: SourceIdentity, guard: ResourceGuard
    ) -> SpeechRiskEvidence: ...


@dataclass(frozen=True)
class BlockEnergies:
    """Per-block RMS of source and engine output for one contiguous range of a file.

    Ranges measured separately (one per parallel chunk) concatenate into the
    whole-file sequence the gate judges, so the verdict does not depend on how
    the file was split.
    """

    source_rms: np.ndarray
    result_rms: np.ndarray
    block_frames: np.ndarray

    @staticmethod
    def concatenate(parts: list["BlockEnergies"]) -> "BlockEnergies":
        return BlockEnergies(
            np.concatenate([p.source_rms for p in parts]),
            np.concatenate([p.result_rms for p in parts]),
            np.concatenate([p.block_frames for p in parts]),
        )


class AudioQualityGate:
    def __init__(self, speech: SpeechRiskAnalyser | None = None):
        self.speech = speech

    @staticmethod
    def block_energies(source: Path, processed: Path, guard: ResourceGuard) -> BlockEnergies:
        """Hard integrity checks plus block energies; raises on corrupt engine output."""
        original_blocks: list[np.ndarray] = []
        output_blocks: list[np.ndarray] = []
        block_frames: list[int] = []
        with sf.SoundFile(source) as original, sf.SoundFile(processed) as output:
            expected_channels = original.channels
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
            while True:
                guard.check()
                before = original.read(BLOCK_FRAMES, dtype="float64", always_2d=True)
                after = output.read(BLOCK_FRAMES, dtype="float64", always_2d=True)
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
                if np.max(original_rms) < 1e-8 and np.max(output_rms) > 1e-4:
                    raise CleanExecutionError(
                        ErrorCode.INVALID_AUDIO, "generated content on silent source"
                    )
                original_blocks.append(original_rms)
                output_blocks.append(output_rms)
                block_frames.append(len(before))
            if frames != original.frames:
                raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "engine decode is incomplete")
        channels = expected_channels
        return BlockEnergies(
            np.array(original_blocks, dtype=np.float64).reshape(-1, channels),
            np.array(output_blocks, dtype=np.float64).reshape(-1, channels),
            np.array(block_frames, dtype=np.int64),
        )

    def summarize(
        self,
        energies: BlockEnergies,
        *,
        source: Path | None = None,
        processed: Path | None = None,
        guard: ResourceGuard | None = None,
        expected_source: SourceIdentity | None = None,
    ) -> ValidationSummary:
        warnings = {"wanted_content_requires_review"}
        intervals: list[ContentWarningInterval] = []
        truncated = False
        source_rms, result_rms = energies.source_rms, energies.result_rms
        # Energy alone cannot tell quiet speech from noise, so the hard rule is
        # file-level; speech preservation is checked with a VAD by the caller.
        total_source = np.sqrt(np.sum(source_rms**2, axis=0))
        total_result = np.sqrt(np.sum(result_rms**2, axis=0))
        active = total_source > 1e-4
        if np.any(active & (total_result < total_source * 0.1)):
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "channel content disappeared")
        # Review hints only for blocks within 12 dB of the loudest block.
        loud = source_rms >= np.maximum(source_rms.max(axis=0, initial=0.0) * 0.25, 1e-4)
        position = 0
        for index, length in enumerate(energies.block_frames.tolist()):
            start = position
            position += length
            judged = loud[index]
            if not np.any(judged):
                continue
            ratio = float(np.min(result_rms[index][judged] / source_rms[index][judged]))
            if ratio >= 0.5:
                continue
            warnings.add("possible_wanted_content_loss")
            code: Literal["possible_wanted_content_loss"] = "possible_wanted_content_loss"
            if intervals and intervals[-1].code == code and intervals[-1].end_frame == start:
                previous = intervals.pop()
                intervals.append(
                    ContentWarningInterval(
                        start_frame=previous.start_frame,
                        end_frame=position,
                        code=code,
                        minimum_rms_ratio=min(previous.minimum_rms_ratio, ratio),
                    )
                )
            elif len(intervals) < 128:
                intervals.append(
                    ContentWarningInterval(
                        start_frame=start,
                        end_frame=position,
                        code=code,
                        minimum_rms_ratio=ratio,
                    )
                )
            else:
                truncated = True
        speech = None
        if self.speech is None:
            warnings.add("speech_activity_unavailable")
        else:
            if expected_source is None or source is None or processed is None or guard is None:
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
            for channel, active_frames in enumerate(speech.source_active_frames):
                target = speech.output_active_frames[
                    channel
                    if len(speech.output_active_frames) == len(speech.source_active_frames)
                    else 0
                ]
                if active_frames and target < active_frames * 0.95:
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

    def evaluate(
        self,
        source: Path,
        processed: Path,
        plan: CleanPlan,
        guard: ResourceGuard,
        *,
        expected_source: SourceIdentity | None = None,
    ) -> ValidationSummary:
        return self.summarize(
            self.block_energies(source, processed, guard),
            source=source,
            processed=processed,
            guard=guard,
            expected_source=expected_source,
        )
