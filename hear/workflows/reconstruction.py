from __future__ import annotations

import uuid
from pathlib import Path

from hear.audio.io import AudioIO
from hear.audio.workspace import AudioWorkspace
from hear.contracts.events import ExecutionEvent, ExecutionEventType
from hear.contracts.jobs import AttemptEnvelope, ReconstructionOperation
from hear.contracts.outcomes import ExecutionOutcome
from hear.services.reconstruction.synthesizer import SpeechSynthesizer
from hear.storage.b2 import B2StorageFactory
from hear.storage.reconstruction import ReconstructionStorageAdapter


class ReconstructionWorkflow:
    def __init__(
        self,
        synthesizer: SpeechSynthesizer,
        audio: AudioIO,
        storage_factory: B2StorageFactory,
        *,
        workspace_root: Path,
    ) -> None:
        self._synthesizer = synthesizer
        self._audio = audio
        self._storage_factory = storage_factory
        self._workspace_root = workspace_root

    async def stream(self, envelope: AttemptEnvelope):
        operation = envelope.operation
        if operation is None:
            raise ValueError("reconstruction_operation_required")
        workspace = AudioWorkspace(
            self._workspace_root / envelope.workspace_namespace,
            envelope.job_id,
            envelope.attempt_id,
        )
        sequence = 1
        try:
            yield self._event(envelope, sequence, "preparing", 0)
            sequence += 1
            source = await self._audio.download_to_wav(
                str(envelope.source.url),
                workspace,
                preserve_channels=True,
            )
            storage = ReconstructionStorageAdapter(self._storage_factory.create(envelope.storage))
            yield self._event(envelope, sequence, "reconstructing", 15)
            sequence += 1
            result = await self._execute(
                envelope,
                operation,
                source,
                storage,
                workspace,
            )
            yield self._event(
                envelope,
                sequence,
                "artifact_prepared",
                90,
                event_type=ExecutionEventType.ARTIFACT_PREPARED,
                data={"reconstructed_audio": result},
            )
            sequence += 1
            outcome = ExecutionOutcome(
                job_id=envelope.job_id,
                attempt_id=envelope.attempt_id,
                track_id=envelope.track_id,
                job_type=envelope.job_type,
                source_revision=envelope.source.revision,
                status="completed",
                artifacts=storage.artifacts,
                result={
                    "job_id": envelope.job_id,
                    "run_id": envelope.run_id,
                    "job_type": envelope.job_type.value,
                    "operation": operation.value,
                    "track_id": envelope.track_id,
                    "requires_approval": True,
                    "reconstructed_audio": result,
                },
            )
            yield self._event(
                envelope,
                sequence,
                "completed",
                100,
                event_type=ExecutionEventType.OUTCOME,
                data={"outcome": outcome.model_dump(mode="json")},
            )
        finally:
            workspace.cleanup()

    async def _execute(
        self,
        envelope: AttemptEnvelope,
        operation: ReconstructionOperation,
        source: Path,
        storage: ReconstructionStorageAdapter,
        workspace: AudioWorkspace,
    ) -> dict:
        same_speaker = bool(envelope.options.get("same_speaker", True))
        if operation in {
            ReconstructionOperation.REPLACE_SEGMENTS,
            ReconstructionOperation.EDIT_TRANSCRIPT,
            ReconstructionOperation.PREVIEW,
        }:
            changes = envelope.options.get("changes")
            if not isinstance(changes, list) or not changes:
                raise ValueError("reconstruction_changes_required")
            result = await self._synthesizer.reconstruct_segments(
                str(source),
                envelope.track_id,
                changes,
                storage,
                same_speaker=same_speaker,
                job_id=envelope.job_id,
                workspace=workspace,
            )
        elif operation == ReconstructionOperation.REBUILD:
            edited_transcript = str(envelope.options.get("edited_transcript") or "").strip()
            if not edited_transcript:
                raise ValueError("edited_transcript_required")
            result = await self._synthesizer.rebuild_track_audio(
                str(source),
                edited_transcript,
                envelope.track_id,
                envelope.job_id,
                storage,
                original_transcript=str(envelope.options.get("original_transcript") or ""),
                workspace=workspace,
            )
        elif operation == ReconstructionOperation.REMOVE_SEGMENTS:
            start = envelope.options.get("segment_start")
            end = envelope.options.get("segment_end")
            if start is None or end is None:
                raise ValueError("segment_interval_required")
            result = await self._synthesizer.remove_segment(
                str(source),
                envelope.track_id,
                float(start),
                float(end),
                storage,
                envelope.job_id,
                workspace=workspace,
            )
        else:
            raise ValueError("unsupported_reconstruction_operation")
        return {
            "b2_key": result.b2_key,
            "audio_url": result.audio_url,
            "duration": result.duration,
            "bucket_name": result.bucket_name or storage.bucket_name,
            "segments": [
                {
                    "segment_start": item.segment_start,
                    "segment_end": item.segment_end,
                    "b2_key": item.b2_key,
                    "audio_url": item.audio_url,
                    "duration": item.duration,
                    "is_deletion": item.is_deletion,
                    "bucket_name": item.bucket_name,
                }
                for item in result.segments
            ],
        }

    @staticmethod
    def _event(
        envelope: AttemptEnvelope,
        sequence: int,
        stage: str,
        progress: float,
        *,
        event_type: ExecutionEventType = ExecutionEventType.STAGE,
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
