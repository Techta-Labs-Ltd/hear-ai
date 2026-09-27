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
from hear.services.categorization.discovery import DiscoverySerialization, DiscoveryService
from hear.services.categorization.service import CategorizationService
from hear.services.moderation.service import ModerationService
from hear.services.transcription.service import TranscriptionService
from hear.storage.b2 import B2StorageFactory


class PipelineProgress:
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


class PipelineWorkflow:
    def __init__(
        self,
        transcriber: TranscriptionService,
        moderator: ModerationService,
        categorizer: CategorizationService,
        discovery: DiscoveryService,
        audio: AudioIO,
        storage_factory: B2StorageFactory,
        native: NativeExecutor,
        *,
        workspace_root: Path,
        bitrate_kbps: int = 96,
    ) -> None:
        self._transcriber = transcriber
        self._moderator = moderator
        self._categorizer = categorizer
        self._discovery = discovery
        self._audio = audio
        self._storage_factory = storage_factory
        self._native = native
        self._workspace_root = workspace_root
        self._bitrate_kbps = bitrate_kbps

    async def stream(self, envelope: AttemptEnvelope):
        workspace = AudioWorkspace(
            self._workspace_root,
            envelope.job_id,
            envelope.attempt_id,
        )
        sequence = 1
        yield self._event(envelope, sequence, ExecutionEventType.STARTED, "preparing", 0)
        sequence += 1
        task: asyncio.Task | None = None
        try:
            source = await self._audio.download_source(
                str(envelope.source.url),
                workspace,
            )
            yield self._event(envelope, sequence, ExecutionEventType.STAGE, "transcribing", 5)
            sequence += 1
            progress = PipelineProgress()
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
                    "transcribing",
                    5 + percent * 0.45,
                )
                sequence += 1
            transcription = await task
            transcript = str(transcription.get("transcript") or "").strip()
            segments = list(transcription.get("segments") or [])
            yield self._event(
                envelope,
                sequence,
                ExecutionEventType.ARTIFACT_PREPARED,
                "transcribing",
                50,
                {"transcription": self._summary(transcription)},
            )
            sequence += 1
            yield self._event(envelope, sequence, ExecutionEventType.STAGE, "moderating", 55)
            sequence += 1
            moderation = await self._moderator.moderate(transcript)
            yield self._event(
                envelope,
                sequence,
                ExecutionEventType.ARTIFACT_PREPARED,
                "moderating",
                62,
                {"moderation": moderation},
            )
            sequence += 1
            categorization = None
            discovery = None
            content_description = None
            if transcript and not moderation.get("flagged"):
                yield self._event(envelope, sequence, ExecutionEventType.STAGE, "categorizing", 65)
                sequence += 1
                categorization = await self._categorizer.categorize(
                    transcript=transcript,
                    segments=segments,
                    max_tags=int(envelope.options.get("max_tags") or 8),
                    per_track_transcripts={envelope.track_id: transcript},
                )
                yield self._event(
                    envelope,
                    sequence,
                    ExecutionEventType.ARTIFACT_PREPARED,
                    "categorizing",
                    73,
                    {"categorization": categorization},
                )
                sequence += 1
                yield self._event(envelope, sequence, ExecutionEventType.STAGE, "discovering", 75)
                sequence += 1
                profile = await self._discovery.build_profile(
                    transcript,
                    content_id=envelope.track_id,
                    track_name=str(envelope.options.get("track_name") or ""),
                    duration_seconds=float(transcription.get("audio_duration") or 0.0) or None,
                    source=envelope.options.get("source"),
                    speaker=envelope.options.get("speaker"),
                    categorization=categorization,
                    prior_description=envelope.options.get("content_description"),
                )
                if profile is not None:
                    discovery = DiscoverySerialization.discovery_to_callback_dict(
                        profile,
                        duration_seconds=float(transcription.get("audio_duration") or 0.0)
                        or None,
                        source=envelope.options.get("source"),
                    )
                    content_description = (
                        DiscoverySerialization.content_description_from_discovery(profile)
                    )
            yield self._event(envelope, sequence, ExecutionEventType.STAGE, "compressing", 82)
            sequence += 1
            mp3_path = workspace.file("delivery.mp3")
            encoded = await self._audio.encode_mp3(
                source,
                mp3_path,
                maximum_kbps=self._bitrate_kbps,
            )
            storage = self._storage_factory.create(envelope.storage)
            key = storage.key(
                "jobs",
                envelope.job_id,
                envelope.attempt_id,
                "delivery.mp3",
            )
            artifact = await self._native.run(
                storage.upload_file,
                mp3_path,
                key,
                sha256=str(encoded["sha256"]),
                content_type="audio/mpeg",
            )
            result = {
                "transcription": transcription,
                "moderation": moderation,
                "categorization": categorization,
                "discovery": discovery,
                "content_description": content_description,
                "flagged": bool(moderation.get("flagged")),
                "compressed_audio": {
                    "audio_url": artifact.audio_url,
                    "b2_key": artifact.object_key,
                    "bucket_name": artifact.bucket_name,
                    "format": "mp3",
                    "bitrate_kbps": int(encoded["bitrate_kbps"]),
                    "size_bytes": artifact.size_bytes,
                },
            }
            manifest_key = storage.key(
                "jobs",
                envelope.job_id,
                envelope.attempt_id,
                "pipeline.json",
            )
            manifest = await self._native.run(
                storage.upload_json,
                result,
                manifest_key,
            )
            outcome = ExecutionOutcome(
                job_id=envelope.job_id,
                attempt_id=envelope.attempt_id,
                track_id=envelope.track_id,
                job_type=envelope.job_type,
                source_revision=envelope.source.revision,
                status="completed",
                artifacts=(artifact, manifest),
                result={
                    "pipeline_manifest": manifest.model_dump(mode="json"),
                    "compressed_audio": result["compressed_audio"],
                    "flagged": result["flagged"],
                    "silent": bool(transcription.get("silent", False)),
                },
            )
            yield self._event(
                envelope,
                sequence,
                ExecutionEventType.OUTCOME,
                "completed",
                100,
                {"outcome": outcome.model_dump(mode="json")},
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
    def _summary(transcription: dict) -> dict:
        return {
            "language": transcription.get("language"),
            "duration": transcription.get("duration"),
            "confidence": transcription.get("confidence"),
            "silent": transcription.get("silent", False),
        }

    @staticmethod
    def _event(
        envelope: AttemptEnvelope,
        sequence: int,
        event_type: ExecutionEventType,
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
