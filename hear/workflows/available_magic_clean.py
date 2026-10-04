from __future__ import annotations

import asyncio
import hashlib
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

from hear.audio.io import AudioIO
from hear.audio.workspace import AudioWorkspace
from hear.contracts.events import ExecutionEvent, ExecutionEventType
from hear.contracts.jobs import AttemptEnvelope, MagicCleanProfile
from hear.contracts.outcomes import ExecutionOutcome
from hear.execution.native import NativeExecutor
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode
from hear.storage.b2 import B2StorageFactory
from hear.workflows.cleaning_progress import CleaningProgress


class AvailableMagicCleanWorkflow:
    def __init__(
        self,
        audio: AudioIO,
        storage_factory: B2StorageFactory,
        native: NativeExecutor,
        *,
        workspace_root: Path,
        timeout_seconds: float,
        model_cleaner=None,
    ) -> None:
        self._audio = audio
        self._storage_factory = storage_factory
        self._native = native
        self._workspace_root = workspace_root
        self._timeout_seconds = timeout_seconds
        self._model_cleaner = model_cleaner

    async def stream(self, envelope: AttemptEnvelope):
        profile = MagicCleanProfile(str(envelope.options.get("profile") or ""))
        workspace = AudioWorkspace(self._workspace_root / envelope.workspace_namespace, envelope.job_id, envelope.attempt_id)
        sequence = 1
        yield self._event(envelope, sequence, "preparing", 0, ExecutionEventType.STARTED)
        sequence += 1
        try:
            if envelope.deadline <= datetime.now(UTC):
                raise CleanExecutionError(ErrorCode.DEADLINE_EXCEEDED, "attempt deadline exceeded")
            cleaner = self._model_cleaner
            if cleaner is None:
                raise CleanExecutionError(
                    ErrorCode.ENGINE_UNAVAILABLE, "magic_clean_model_engine_unavailable"
                )
            sound = envelope.options.get("sound_cleanup", {})
            if sound.get("enabled") and not getattr(cleaner, "sound_cleanup_available", False):
                raise CleanExecutionError(
                    ErrorCode.ENGINE_UNAVAILABLE, "sound_cleanup_not_provisioned"
                )
            if sound.get("preview_overlaps") and not getattr(
                cleaner, "overlap_preview_available", False
            ):
                raise CleanExecutionError(
                    ErrorCode.ENGINE_UNAVAILABLE, "overlap_separator_not_provisioned"
                )
            if envelope.options.get("reduce_stationary_noise") and not getattr(
                self._model_cleaner, "sound_cleanup_available", False
            ):
                raise CleanExecutionError(
                    ErrorCode.ENGINE_UNAVAILABLE, "background_analyser_not_provisioned"
                )
            source = await self._audio.download_source(str(envelope.source.url), workspace)
            source_digest = await self._native.run(self._sha256, source)
            if envelope.source.file_sha256 and source_digest != envelope.source.file_sha256:
                raise CleanExecutionError(
                    ErrorCode.SOURCE_MISMATCH, "downloaded source digest mismatch"
                )
            yield self._event(envelope, sequence, "processing", 20, ExecutionEventType.STAGE)
            sequence += 1
            cleaner = self._model_cleaner
            if cleaner is None or profile.value not in getattr(
                cleaner, "supported_profiles", (cleaner.profile,)
            ):
                raise CleanExecutionError(
                    ErrorCode.ENGINE_UNAVAILABLE, "magic_clean_model_engine_unavailable"
                )
            cancellation = threading.Event()
            progress = CleaningProgress()
            task = asyncio.create_task(
                self._native.run_cancellable(
                    cleaner.clean,
                    source,
                    workspace.path,
                    envelope.options,
                    envelope.deadline,
                    self._timeout_seconds,
                    cancelled=cancellation,
                    progress=progress.publish,
                )
            )
            try:
                while not task.done():
                    try:
                        stage, percent = await asyncio.wait_for(progress.queue.get(), timeout=0.2)
                    except TimeoutError:
                        continue
                    yield self._event(envelope, sequence, stage, percent, ExecutionEventType.STAGE)
                    sequence += 1
                clean_report = await task
            finally:
                cancellation.set()
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            engine = cleaner.engine
            delivery = workspace.file("delivery_audio.mp3")
            if (
                not isinstance(clean_report, dict)
                or not delivery.is_file()
                or clean_report.get("technical_validation") != "passed"
                or clean_report.get("profile") != profile.value
                or clean_report.get("engine") != engine
            ):
                raise CleanExecutionError(
                    ErrorCode.INVALID_AUDIO, "magic_clean_validated_delivery_missing"
                )
            delivery_digest = await self._native.run(self._sha256, delivery)
            encoded = {
                "sha256": delivery_digest,
                "duration_seconds": clean_report["duration_seconds"],
            }
            yield self._event(envelope, sequence, "uploading", 85, ExecutionEventType.STAGE)
            sequence += 1
            storage = self._storage_factory.create(envelope.storage)
            delivery_artifact = await self._native.run(
                storage.upload_file,
                delivery,
                storage.key("jobs", envelope.job_id, envelope.attempt_id, "delivery_audio.mp3"),
                sha256=str(encoded["sha256"]),
                content_type="audio/mpeg",
            )
            validation = {
                **clean_report,
                "source_sha256": source_digest,
                "source_revision": envelope.source.revision,
                "engine": engine,
                "profile": profile.value,
                "duration_seconds": encoded["duration_seconds"],
                "delivery_sha256": encoded["sha256"],
                "status": "passed",
            }
            validation_artifact = await self._native.run(
                storage.upload_json,
                validation,
                storage.key("jobs", envelope.job_id, envelope.attempt_id, "validation.json"),
            )
            outcome = ExecutionOutcome(
                job_id=envelope.job_id,
                attempt_id=envelope.attempt_id,
                track_id=envelope.track_id,
                job_type=envelope.job_type,
                source_revision=envelope.source.revision,
                status="completed",
                artifacts=(delivery_artifact, validation_artifact),
                result={
                    "profile": profile.value,
                    "engine": engine,
                    "requires_approval": True,
                    "delivery_audio": delivery_artifact.model_dump(mode="json"),
                    "validation": validation,
                },
            )
            yield self._event(
                envelope,
                sequence,
                "completed",
                100,
                ExecutionEventType.OUTCOME,
                {"outcome": outcome.model_dump(mode="json")},
            )
        except CleanExecutionError as error:
            outcome = ExecutionOutcome(
                job_id=envelope.job_id,
                attempt_id=envelope.attempt_id,
                track_id=envelope.track_id,
                job_type=envelope.job_type,
                source_revision=envelope.source.revision,
                status="failed",
                error_code=error.code.value,
                result={"message": str(error)},
            )
            yield self._event(
                envelope,
                sequence,
                "failed",
                100,
                ExecutionEventType.OUTCOME,
                {"outcome": outcome.model_dump(mode="json")},
            )
        finally:
            workspace.cleanup()

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _event(
        envelope: AttemptEnvelope,
        sequence: int,
        stage: str,
        progress: float,
        event_type: ExecutionEventType,
        data: dict | None = None,
    ) -> ExecutionEvent:
        return ExecutionEvent(
            event_id=str(uuid.uuid4()),
            job_id=envelope.job_id,
            attempt_id=envelope.attempt_id,
            track_id=envelope.track_id,
            job_type=envelope.job_type,
            source_revision=envelope.source.revision,
            sequence=sequence,
            event=event_type,
            stage=stage,
            progress_pct=progress,
            data=data or {},
        )
