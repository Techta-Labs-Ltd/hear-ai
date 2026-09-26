from __future__ import annotations

import asyncio
import hashlib
import uuid
from pathlib import Path

from hear.audio.io import AudioIO
from hear.audio.workspace import AudioWorkspace
from hear.contracts.events import ExecutionEvent, ExecutionEventType
from hear.contracts.jobs import AttemptEnvelope
from hear.contracts.outcomes import ExecutionOutcome
from hear.execution.native import NativeExecutor
from hear.services.categorization.discovery import DiscoveryService, DiscoverySupport
from hear.services.categorization.service import CategorizationService
from hear.services.moderation.service import ModerationService
from hear.services.transcription.service import TranscriptionService
from hear.storage.b2 import B2StorageFactory
from hear.utils.audio import convert_wav_file_to_mp3, delivery_bitrate_kbps


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
        yield self._event(
            envelope,
            sequence,
            ExecutionEventType.STARTED,
            stage="preparing",
            progress=0,
        )
        sequence += 1
        transcription_task: asyncio.Task | None = None
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
            progress = PipelineProgress()
            transcription_task = asyncio.create_task(
                self._transcriber.transcribe_file(
                    str(source),
                    job_id=envelope.job_id,
                    run_id=envelope.run_id,
                    track_id=envelope.track_id,
                    progress=progress,
                )
            )
            while not transcription_task.done():
                try:
                    percent = await asyncio.wait_for(progress.next(), timeout=1.0)
                except TimeoutError:
                    continue
                yield self._event(
                    envelope,
                    sequence,
                    ExecutionEventType.PROGRESS,
                    stage="transcribing",
                    progress=5 + percent * 0.45,
                )
                sequence += 1
            transcription = await transcription_task
            transcript = str(transcription.get("transcript") or "").strip()
            segments = list(transcription.get("segments") or [])
            yield self._event(
                envelope,
                sequence,
                ExecutionEventType.STAGE,
                stage="moderating",
                progress=52,
            )
            sequence += 1
            moderation = (
                await self._moderator.moderate(transcript)
                if transcript
                else self._empty_moderation()
            )
            categorization = None
            discovery = None
            content_description = None
            if transcript and not moderation.get("flagged"):
                yield self._event(
                    envelope,
                    sequence,
                    ExecutionEventType.STAGE,
                    stage="categorizing",
                    progress=62,
                )
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
                    ExecutionEventType.STAGE,
                    stage="discovering",
                    progress=74,
                )
                sequence += 1
                profile = await self._discovery.build_profile(
                    transcript,
                    content_id=envelope.track_id,
                    track_name=str(envelope.options.get("track_name") or ""),
                    duration_seconds=self._duration(transcription),
                    source=str(envelope.options.get("source") or "") or None,
                    speaker=str(envelope.options.get("speaker") or "") or None,
                    categorization=categorization,
                    prior_description=str(
                        envelope.options.get("content_description") or ""
                    )
                    or None,
                )
                discovery, content_description = DiscoverySupport.discovery_result_bundle(
                    profile,
                    duration_seconds=self._duration(transcription),
                    source=str(envelope.options.get("source") or "") or None,
                )
            yield self._event(
                envelope,
                sequence,
                ExecutionEventType.STAGE,
                stage="compressing",
                progress=84,
            )
            sequence += 1
            storage = self._storage_factory.create(envelope.storage)
            mp3_path = await self._encode(source, envelope, workspace)
            digest = await self._native.run(self._sha256, mp3_path)
            key = storage.key(
                "jobs",
                envelope.job_id,
                envelope.attempt_id,
                "source.mp3",
            )
            audio_artifact = await self._native.run(
                storage.upload_file,
                mp3_path,
                key,
                sha256=digest,
                content_type="audio/mpeg",
            )
            result_payload = {
                "transcription": transcription,
                "moderation": moderation,
                "categorization": categorization,
                "discovery": discovery,
                "content_description": content_description,
                "flagged": bool(moderation.get("flagged")),
                "compressed_audio": {
                    "audio_url": audio_artifact.audio_url,
                    "b2_key": audio_artifact.object_key,
                    "bucket_name": audio_artifact.bucket_name,
                    "format": "mp3",
                    "size_bytes": audio_artifact.size_bytes,
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
                result_payload,
                manifest_key,
            )
            outcome = ExecutionOutcome(
                job_id=envelope.job_id,
                attempt_id=envelope.attempt_id,
                track_id=envelope.track_id,
                job_type=envelope.job_type,
                source_revision=envelope.source.revision,
                status="completed",
                artifacts=(audio_artifact, manifest),
                result={
                    "pipeline_manifest": manifest.model_dump(mode="json"),
                    "compressed_audio": result_payload["compressed_audio"],
                    "flagged": result_payload["flagged"],
                    "silent": bool(transcription.get("silent", False)),
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
            if transcription_task is not None and not transcription_task.done():
                transcription_task.cancel()
                try:
                    await transcription_task
                except asyncio.CancelledError:
                    pass
            workspace.cleanup()

    async def _encode(
        self,
        source: Path,
        envelope: AttemptEnvelope,
        workspace: AudioWorkspace,
    ) -> Path:
        bitrate = await self._native.run(
            delivery_bitrate_kbps,
            str(source),
            maximum_kbps=self._bitrate_kbps,
        )
        generated = await convert_wav_file_to_mp3(
            str(source),
            bitrate_kbps=bitrate,
            job_id=envelope.job_id,
            run_id=envelope.run_id,
            track_id=envelope.track_id,
            purpose="pipeline_output",
        )
        target = workspace.file("source.mp3")
        await self._native.run(Path(generated).replace, target)
        return target

    @staticmethod
    def _duration(transcription: dict) -> float | None:
        value = transcription.get("audio_duration", transcription.get("duration"))
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _empty_moderation() -> dict:
        return {
            "flagged": False,
            "severity": "none",
            "intent": "safe",
            "reason": "",
            "flagged_categories": [],
            "blocked_words_found": [],
        }

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
