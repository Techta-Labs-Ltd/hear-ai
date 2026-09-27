from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from hear.audio.io import AudioIO
from hear.audio.workspace import AudioWorkspace
from hear.contracts.events import ExecutionEvent, ExecutionEventType
from hear.contracts.jobs import AttemptEnvelope, MagicCleanProfile
from hear.contracts.outcomes import ExecutionOutcome
from hear.execution.native import NativeExecutor
from hear.services.magic_clean.contracts import CleanExecutionError
from hear.storage.b2 import B2StorageFactory


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
        workspace = AudioWorkspace(self._workspace_root, envelope.job_id, envelope.attempt_id)
        sequence = 1
        yield self._event(envelope, sequence, "preparing", 0, ExecutionEventType.STARTED)
        sequence += 1
        try:
            source = await self._audio.download_source(str(envelope.source.url), workspace)
            yield self._event(envelope, sequence, "processing", 20, ExecutionEventType.STAGE)
            sequence += 1
            master = workspace.file("cleaned_master.flac")
            cleaner = self._model_cleaner
            if cleaner is None or cleaner.profile != profile.value:
                raise RuntimeError("magic_clean_model_engine_unavailable")
            await self._native.run(
                cleaner.clean,
                source,
                master,
                workspace.path,
                envelope.options,
                envelope.deadline,
                self._timeout_seconds,
            )
            engine = cleaner.engine
            delivery = workspace.file("delivery_audio.mp3")
            encoded = await self._audio.encode_mp3(master, delivery, maximum_kbps=96)
            yield self._event(envelope, sequence, "uploading", 85, ExecutionEventType.STAGE)
            sequence += 1
            storage = self._storage_factory.create(envelope.storage)
            master_digest = await self._native.run(self._sha256, master)
            master_artifact = await self._native.run(
                storage.upload_file,
                master,
                storage.key("jobs", envelope.job_id, envelope.attempt_id, "cleaned_master.flac"),
                sha256=master_digest,
                content_type="audio/flac",
            )
            delivery_artifact = await self._native.run(
                storage.upload_file,
                delivery,
                storage.key("jobs", envelope.job_id, envelope.attempt_id, "delivery_audio.mp3"),
                sha256=str(encoded["sha256"]),
                content_type="audio/mpeg",
            )
            validation = {
                "engine": engine,
                "profile": profile.value,
                "duration_seconds": encoded["duration_seconds"],
                "master_sha256": master_digest,
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
                artifacts=(master_artifact, delivery_artifact, validation_artifact),
                result={
                    "profile": profile.value,
                    "engine": engine,
                    "requires_approval": True,
                    "cleaned_master": master_artifact.model_dump(mode="json"),
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
