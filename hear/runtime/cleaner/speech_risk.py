import hashlib
from pathlib import Path

import soundfile as sf

from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.runtime.cleaner.speech_activity import CpuSpeechActivity, SpeechActivityReport
from hear.services.magic_clean.contracts import (
    CleanExecutionError,
    ErrorCode,
    SourceIdentity,
    SpeechLossInterval,
    SpeechRiskEvidence,
)


class SpeechRiskComparison:
    POLICY = "speech-risk-v1-missing100ms-active-ratio0.5-no-truncated-interval-inference"
    DIGEST = hashlib.sha256(POLICY.encode()).hexdigest()

    def __init__(self, analyser: CpuSpeechActivity):
        self.analyser = analyser

    def evaluate(
        self, source: Path, processed: Path, expected: SourceIdentity, guard: ResourceGuard
    ) -> SpeechRiskEvidence:
        try:
            return self._evaluate(source, processed, expected, guard)
        except CleanExecutionError:
            raise
        except (OSError, ValueError, RuntimeError):
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "speech comparison failed") from None

    def _evaluate(
        self, source: Path, processed: Path, expected: SourceIdentity, guard: ResourceGuard
    ) -> SpeechRiskEvidence:
        before = self.analyser.scan(source, expected, guard)
        guard.check()
        if (
            processed.is_symlink()
            or not processed.is_file()
            or not processed.resolve().is_relative_to(guard.workspace.resolve())
        ):
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "speech output outside workspace")
        digest = hashlib.sha256()
        size = 0
        with processed.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                guard.check()
                size += len(chunk)
                if size > guard.budget.max_input_bytes:
                    raise CleanExecutionError(
                        ErrorCode.RESOURCE_EXHAUSTED, "speech input too large"
                    )
                digest.update(chunk)
        with sf.SoundFile(processed) as audio:
            output_identity = SourceIdentity(
                revision_id="attempt-local-validation",
                media_id="attempt-local-validation",
                object_key="attempt-local/processed.wav",
                object_version="local",
                sha256=digest.hexdigest(),
                size_bytes=size,
                sample_rate=audio.samplerate,
                channels=audio.channels,
                frames=audio.frames,
            )
        after = self.analyser.scan(processed, output_identity, guard)
        guard.check()
        return self.compare(before, after, expected)

    @classmethod
    def compare(
        cls, before: SpeechActivityReport, after: SpeechActivityReport, expected: SourceIdentity
    ) -> SpeechRiskEvidence:
        if (
            before.source_sha256 != expected.sha256
            or before.policy_sha256 != after.policy_sha256
            or (before.frames, before.sample_rate, before.channels)
            != (expected.frames, expected.sample_rate, expected.channels)
            or (after.frames, after.sample_rate) != (expected.frames, expected.sample_rate)
            or after.channels not in (1, before.channels)
        ):
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "speech evidence grid mismatch")
        for report in (before, after):
            if (
                len(report.active_frames) != report.channels
                or any(not 0 <= count <= expected.frames for count in report.active_frames)
                or any(
                    not 0 <= item.channel < report.channels
                    or not 0 <= item.start_frame < item.end_frame <= expected.frames
                    for item in report.intervals
                )
            ):
                raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "invalid speech evidence")
        incomplete = before.intervals_truncated or after.intervals_truncated
        losses = []
        if not incomplete:
            minimum = max(1, (expected.sample_rate + 9) // 10)
            for interval in before.intervals:
                channel = interval.channel if after.channels == before.channels else 0
                cursor = interval.start_frame
                targets = sorted(
                    (v for v in after.intervals if v.channel == channel),
                    key=lambda v: v.start_frame,
                )
                for target in targets:
                    if target.end_frame <= cursor:
                        continue
                    if target.start_frame >= interval.end_frame:
                        break
                    if target.start_frame - cursor >= minimum:
                        losses.append(
                            SpeechLossInterval(
                                channel=interval.channel,
                                start_frame=cursor,
                                end_frame=target.start_frame,
                            )
                        )
                    cursor = max(cursor, target.end_frame)
                    if cursor >= interval.end_frame:
                        break
                if interval.end_frame - cursor >= minimum:
                    losses.append(
                        SpeechLossInterval(
                            channel=interval.channel,
                            start_frame=cursor,
                            end_frame=interval.end_frame,
                        )
                    )
                if len(losses) > 128:
                    incomplete = True
                    losses = losses[:128]
                    break
        return SpeechRiskEvidence(
            source_sha256=before.source_sha256,
            output_sha256=after.source_sha256,
            analysis_sha256=before.policy_sha256,
            comparison_sha256=cls.DIGEST,
            source_active_frames=before.active_frames,
            output_active_frames=after.active_frames,
            source_loss_intervals=tuple(losses),
            evidence_truncated=incomplete,
        )
