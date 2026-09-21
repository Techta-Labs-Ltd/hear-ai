"""Shared attempt execution, isolated from legacy service imports during migration.

Ingress supplies an authorizer and either a pinned local source or a scoped
versioned source stager. Downloading occurs only after authorization/admission.
No profile selection, business persistence, retry or approval happens here.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import soundfile as sf

from hear.runtime.cleaner.metrics import StageTimings
from hear.runtime.cleaner.model_registry import EngineRegistry
from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.runtime.cleaner.s3_verification import S3SourceStager
from hear.runtime.cleaner.sam_pipeline import SamSeparationPipeline
from hear.runtime.cleaner.worker_lease import WorkerLease
from hear.services.magic_clean.artifacts import ArtifactWriter, LocalArtifact, PublishedBundle
from hear.services.magic_clean.contracts import (
    AttemptTicket,
    CleanExecutionError,
    CleanPlan,
    ErrorCode,
)
from hear.services.magic_clean.engines.base import EngineSession
from hear.services.magic_clean.inspection import SourceInspector
from hear.services.magic_clean.mastering import AudioMasteringService
from hear.services.magic_clean.quality import AudioQualityGate


class AttemptAuthorizer(Protocol):
    def verify(self, ticket: AttemptTicket) -> None:
        """Verify transport identity, ticket scope/fence/grants and deadline or raise."""
        ...


class ProgressSink(Protocol):
    def transition(self, ticket: AttemptTicket, stage: str) -> None:
        """Persist compact state transitions without exposing private paths/grants."""
        ...


@dataclass(frozen=True)
class ExecutionContext:
    ticket: AttemptTicket
    source: Path
    guard: ResourceGuard
    authorizer: AttemptAuthorizer
    progress: ProgressSink
    timings: StageTimings = field(default_factory=StageTimings)

    def check(self) -> None:
        self.guard.bind_deadline(self.ticket.deadline)
        self.guard.check()
        if datetime.now(UTC) >= self.ticket.deadline:
            raise CleanExecutionError(ErrorCode.DEADLINE_EXCEEDED, "attempt deadline exceeded")


class PublishedExecutionError(CleanExecutionError):
    """Typed execution failure with a verified terminal reference for ingress.

    Transport can send this bundle's compact reference without treating it as
    a successful candidate. The original error remains available as __cause__.
    """

    def __init__(
        self, code: ErrorCode, bundle: PublishedBundle, *, worker_restart_required: bool = False
    ):
        super().__init__(
            code,
            "cleaner attempt failed; terminal manifest published",
            worker_restart_required=worker_restart_required,
        )
        self.bundle = bundle

    def __reduce__(self):
        return type(self), (self.code, self.bundle), self.__dict__


class CleanExecutor:
    def __init__(
        self,
        inspector: SourceInspector,
        registry: EngineRegistry,
        quality: AudioQualityGate,
        masterer: AudioMasteringService,
        artifacts: ArtifactWriter,
        worker_lease: WorkerLease,
    ):
        self.inspector = inspector
        self.registry = registry
        self.quality = quality
        self.masterer = masterer
        self.artifacts = artifacts
        self.worker_lease = worker_lease

    @staticmethod
    def _close_session(session: EngineSession, primary_code: ErrorCode) -> None:
        failed = False
        try:
            session.close()
        except Exception:
            # Unknown native/model teardown state must never permit worker reuse.
            # Do not expose teardown diagnostics or replace cancellation/deadline
            # with an apparently retryable processing failure.
            failed = True
        if failed:
            raise CleanExecutionError(
                primary_code,
                "engine session cleanup failed; worker requires process restart",
                worker_restart_required=True,
            ) from None

    @staticmethod
    def _sample(processed: Path, destination: Path, context: ExecutionContext) -> Path:
        interval = context.ticket.sample
        if interval is None:
            return processed
        if destination.exists():
            raise CleanExecutionError(ErrorCode.ARTIFACT_CONFLICT, "sample output already exists")
        with sf.SoundFile(processed) as source:
            source.seek(interval.start_frame)
            remaining = interval.end_frame - interval.start_frame
            with sf.SoundFile(
                destination,
                "w",
                samplerate=source.samplerate,
                channels=source.channels,
                format="RF64",
                subtype="FLOAT",
            ) as output:
                while remaining:
                    context.check()
                    data = source.read(min(32768, remaining), dtype="float32", always_2d=True)
                    if not len(data):
                        raise CleanExecutionError(
                            ErrorCode.INVALID_AUDIO, "sample source truncated"
                        )
                    output.write(data)
                    remaining -= len(data)
        return destination

    def execute(
        self, plan: CleanPlan, context: ExecutionContext, *, stager: S3SourceStager | None = None
    ) -> PublishedBundle:
        context.authorizer.verify(context.ticket)
        context.check()
        lane = "cpu" if plan.runtime.engine == "noise_profile" else "gpu"
        if plan != context.ticket.plan:
            raise CleanExecutionError(
                ErrorCode.ARTIFACT_CONFLICT, "plan differs from authorized attempt"
            )
        with self.worker_lease.attempt(lane):
            context.timings.reset()
            return self._execute_admitted(plan, context, stager)

    def _execute_admitted(
        self, plan: CleanPlan, context: ExecutionContext, stager: S3SourceStager | None
    ) -> PublishedBundle:
        try:
            if stager is not None:
                context.progress.transition(context.ticket, "downloading")
                with context.timings.measure("download"):
                    stager.stage(context.ticket, context.source, context.guard)
                context.check()
            return self._execute_verified(plan, context)
        except CleanExecutionError as error:
            if error.worker_restart_required:
                self.worker_lease.mark_unhealthy()
            # Unknown upload acceptance, stale authorization and cancelled/expired
            # attempts require reconciliation, not a competing terminal upload.
            if error.code not in (
                ErrorCode.PROCESS_FAILED,
                ErrorCode.INVALID_AUDIO,
                ErrorCode.SOURCE_MISMATCH,
                ErrorCode.ENGINE_UNAVAILABLE,
                ErrorCode.RESOURCE_EXHAUSTED,
            ):
                raise
            try:
                context.check()
                context.authorizer.verify(context.ticket)
                with context.timings.measure("upload"):
                    bundle = self.artifacts.publish_failure(
                        context.ticket, error.code, context.guard
                    )
            except Exception:
                # Finalization must not hide the original typed processing error.
                raise error from None
            raise PublishedExecutionError(
                error.code, bundle, worker_restart_required=error.worker_restart_required
            ) from error

    def _execute_verified(self, plan: CleanPlan, context: ExecutionContext) -> PublishedBundle:
        ticket = context.ticket
        guard = context.guard
        context.progress.transition(ticket, "inspecting")
        with context.timings.measure("inspection"):
            inspected = self.inspector.inspect(context.source, ticket.input, guard)
        # Reserve source, processed PCM, sample, lossless/delivery and validation headroom.
        guard.preflight_pcm(
            inspected.frames,
            inspected.channels,
            copies=6,
            output_bytes=ticket.input.size_bytes + 262144,
        )
        if plan.runtime.engine == "sam_audio_small":
            SamSeparationPipeline.preflight(inspected.frames, inspected.sample_rate, guard)
        context.check()
        self.worker_lease.assert_owned("cpu" if plan.runtime.engine == "noise_profile" else "gpu")
        with context.timings.measure("loading"):
            engine = self.registry.load(
                plan,
                frames=inspected.frames,
                size_bytes=ticket.input.size_bytes,
                sample_rate=inspected.sample_rate,
                channels=inspected.channels,
            )
        processed = guard.workspace / "processed.wav"
        if processed.exists():
            raise CleanExecutionError(
                ErrorCode.ARTIFACT_CONFLICT, "processed output already exists"
            )
        context.progress.transition(ticket, "processing")
        with context.timings.measure("loading"):
            session = engine.open_session(plan, guard)
        primary_code = ErrorCode.PROCESS_FAILED
        try:
            with context.timings.measure("inference"):
                session.process(context.source, processed, plan, guard)
                context.check()
        except CleanExecutionError as error:
            primary_code = error.code
            raise
        finally:
            with context.timings.measure("cleanup"):
                self._close_session(session, primary_code)
        context.progress.transition(ticket, "validating")
        with context.timings.measure("validation"):
            validation = self.quality.evaluate(
                context.source, processed, plan, guard, expected_source=ticket.input
            )
        sample = self._sample(processed, guard.workspace / "sample.wav", context)
        context.progress.transition(ticket, "mastering")
        with context.timings.measure("mastering"):
            mastered = self.masterer.master(sample, plan, guard)
        context.check()
        report = guard.workspace / "validation_report.json"
        payload = {
            "stage_seconds_before_publication": context.timings.snapshot(),
            "validation": validation.model_dump(mode="json"),
            "source": asdict(inspected),
            "master": {
                "gain_db": mastered.gain_db,
                "processing_rate": mastered.processing_rate,
                "delivery_rate": mastered.delivery_rate,
                "frames": mastered.frames,
                "channels": mastered.channels,
                "bit_depth": mastered.bit_depth,
                "dither_policy": mastered.dither_policy,
                "lossless": asdict(mastered.master_measurement),
                "delivery": asdict(mastered.delivery_measurement),
            },
        }
        with report.open("x", encoding="utf-8") as output:
            json.dump(payload, output, sort_keys=True, allow_nan=False)
        context.check()
        context.authorizer.verify(ticket)
        context.progress.transition(ticket, "uploading")
        with context.timings.measure("upload"):
            return self.artifacts.publish(
                ticket,
                (
                    LocalArtifact("cleaned_master", mastered.master),
                    LocalArtifact("delivery_audio", mastered.delivery),
                    LocalArtifact("validation_report", report),
                ),
                validation,
                guard,
            )
