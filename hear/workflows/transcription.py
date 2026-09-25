from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from hear.audio.io import AudioIO
from hear.audio.workspace import AudioWorkspace
from hear.contracts.events import ExecutionEvent, ExecutionEventType
from hear.contracts.jobs import AttemptEnvelope
from hear.contracts.outcomes import ExecutionOutcome
from hear.execution.native import NativeExecutor
from hear.services.transcription.service import TranscriptionService
from hear.storage.b2 import B2StorageFactory


class TranscriptionProgress:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[float] = asyncio.Queue(maxsize=4)

    async def publish(self, progress: float) -> None:
        value = max(0.0, min(float(progress), 100.0))
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        self._queue.put_nowait(value)

    async def next(self) -> float:
        return await self._queue.get()


class TranscriptionWorkflow:
    def __init__(
        self,
        transcriber: TranscriptionService,
        audio: AudioIO,
        storage_factory: B2StorageFactory,
        native: NativeExecutor,
        *,
        workspace_root: Path,
    ) -> None:
        self._transcriber = transcriber
        self._audio = audio
        self._storage_factory = storage_factory
        self._native = native
        self._workspace_root = workspace_root

    async def stream(self, envelope: AttemptEnvelope):
        workspace = AudioWorkspace(
            self._workspace_root,
            envelope.job_id,
            envelope.attempt_id,
        )
        sequence = 1
        yield self._event(
            envelope,
            sequence,
            ExecutionEventType.STARTED,
            stage="preparing",
            progress=0,
        )
        sequence += 1
        task: asyncio.Task | None = None
        try:
            source = await self._audio.download_to_wav(
                str(envelope.source.url),
                workspace,
                preserve_channels=True,
            )
            yield self._event(
                envelope,
                sequence,
                ExecutionEventType.STAGE,
                stage="transcribing",
                progress=5,
            )
            sequence += 1
            progress = TranscriptionProgress()
            task = asyncio.create_task(
                self._transcriber.transcribe_file(
                    str(source),
                    job_id=envelope.job_id,
                    run_id=envelope.run_id,
                    track_id=envelope.track_id,
                    progress=progress,
                )
            )
            while not task.done():
                try:
                    percent = await asyncio.wait_for(progress.next(), timeout=1.0)
                except TimeoutError:
                    continue
                yield self._event(
                    envelope,
                    sequence,
                    ExecutionEventType.PROGRESS,
                    stage="transcribing",
                    progress=5 + percent * 0.85,
                )
                sequence += 1
            transcription = await task
            storage = self._storage_factory.create(envelope.storage)
            key = storage.key(
                "jobs",
                envelope.job_id,
                envelope.attempt_id,
                "transcription.json",
            )
            artifact = await self._native.run(
                storage.upload_json,
                transcription,
                key,
            )
            outcome = ExecutionOutcome(
                job_id=envelope.job_id,
                attempt_id=envelope.attempt_id,
                track_id=envelope.track_id,
                job_type=envelope.job_type,
                source_revision=envelope.source.revision,
                status="completed",
                artifacts=(artifact,),
                result={
                    "transcription_manifest": artifact.model_dump(mode="json"),
                    "language": transcription.get("language"),
                    "duration": transcription.get("duration"),
                    "confidence": transcription.get("confidence"),
                    "silent": transcription.get("silent", False),
                },
            )
            yield self._event(
                envelope,
                sequence,
                ExecutionEventType.OUTCOME,
                stage="completed",
                progress=100,
                data={"outcome": outcome.model_dump(mode="json")},
            )
        finally:
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            workspace.cleanup()

    @staticmethod
    def _event(
        envelope: AttemptEnvelope,
        sequence: int,
        event_type: ExecutionEventType,
        *,
        stage: str,
        progress: float,
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
            progress_pct=round(progress, 2),
            data=data or {},
        )