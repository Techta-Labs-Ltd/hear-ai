from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from hear.audio.io import AudioIO
from hear.audio.source_integrity import SourceIntegrity
from hear.audio.workspace import AudioWorkspace
from hear.contracts.events import ExecutionEvent, ExecutionEventType
from hear.contracts.jobs import AttemptEnvelope
from hear.contracts.outcomes import ExecutionOutcome
from hear.execution.native import NativeExecutor
from hear.services.categorization.audio_content import AudioContentService
from hear.services.categorization.discovery import DiscoverySerialization, DiscoveryService
from hear.services.categorization.service import CategorizationService
from hear.services.moderation.service import ModerationService
from hear.services.transcription.service import TranscriptionService


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
        native: NativeExecutor,
        *,
        workspace_root: Path,
        audio_content: AudioContentService | None = None,
    ) -> None:
        self._transcriber = transcriber
        self._moderator = moderator
        self._categorizer = categorizer
        self._discovery = discovery
        self._audio = audio
        self._native = native
        self._workspace_root = workspace_root
        self._audio_content = audio_content or AudioContentService(None, native)

    async def stream(self, envelope: AttemptEnvelope):
        workspace = AudioWorkspace(
            self._workspace_root / envelope.workspace_namespace,
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
            await self._native.run(SourceIntegrity.verify, source, envelope.source.file_sha256)
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
            audio_content = (
                AudioContentService.result_for("silence")
                if transcription.get("silent")
                else await self._audio_content.analyse(
                    str(source),
                    float(transcription.get("audio_duration") or 0.0),
                    has_speech=bool(transcript),
                )
            )
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
            max_tags = int(envelope.options.get("max_tags") or 8)
            # Moderation adds a review flag; it never withholds tags or discovery, so an
            # admin who clears a flag does not have to re-run the analysis.
            if transcript or audio_content["music"]:
                yield self._event(envelope, sequence, ExecutionEventType.STAGE, "categorizing", 65)
                sequence += 1
                if transcript:
                    categorization = await self._categorizer.categorize(
                        transcript=transcript,
                        segments=segments,
                        max_tags=max_tags,
                        per_track_transcripts={envelope.track_id: transcript},
                    )
                if audio_content["music"]:
                    categorization = self._categorizer.with_song(
                        categorization,
                        confidence=audio_content["music_share"],
                        max_tags=max_tags,
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
            if transcript:
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
            # The pipeline analyses the source; it does not re-encode or store audio.
            # Everything the backend needs travels in the outcome itself.
            outcome = ExecutionOutcome(
                job_id=envelope.job_id,
                attempt_id=envelope.attempt_id,
                track_id=envelope.track_id,
                job_type=envelope.job_type,
                backend_id=envelope.backend_id,
                source_revision=envelope.source.revision,
                status="completed",
                artifacts=(),
                result={
                    "transcription": transcription,
                    "moderation": moderation,
                    "categorization": categorization,
                    "discovery": discovery,
                    "content_description": content_description,
                    "flagged": bool(moderation.get("flagged")),
                    "silent": bool(transcription.get("silent", False)),
                    "no_speech": not transcript,
                    "audio_content": audio_content,
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
            "no_speech": transcription.get("no_speech", False),
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
            backend_id=envelope.backend_id,
            source_revision=envelope.source.revision,
            sequence=sequence,
            event=event_type,
            stage=stage,
            progress_pct=round(progress, 2),
            data=data or {},
        )
