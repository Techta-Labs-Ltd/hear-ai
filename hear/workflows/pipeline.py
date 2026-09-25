from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path

from hear.audio.io import AudioIO
from hear.audio.workspace import AudioWorkspace
from hear.contracts.events import ExecutionEvent, ExecutionEventType
from hear.contracts.jobs import AttemptEnvelope
from hear.contracts.outcomes import ExecutionOutcome
from hear.execution.native import NativeExecutor
from hear.models.discovery import DiscoverySerialization
from hear.services.categorization.discovery import DiscoveryService, DiscoverySupport
from hear.services.categorization.service import CategorizationService
from hear.services.moderation.service import ModerationService
from hear.services.transcription.service import TranscriptionService
from hear.storage.b2 import B2StorageFactory
from hear.utils.audio import convert_wav_file_to_mp3, delivery_bitrate_kbps, probe_audio
from hear.utils.processing_context import effective_transcript_text
from hear.utils.transcript_diff import correct_whisper_mishearings, restore_punctuation_from_edit


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
        maximum_bitrate_kbps: int = 96,
    ) -> None:
        self._transcriber = transcriber
        self._moderator = moderator
        self._categorizer = categorizer
        self._discovery = discovery
        self._audio = audio
        self._storage_factory = storage_factory
        self._native = native
        self._workspace_root = workspace_root
        self._maximum_bitrate_kbps = maximum_bitrate_kbps

    async def stream(self, envelope: AttemptEnvelope):
        workspace = AudioWorkspace(
            self._workspace_root,
            envelope.job_id,
            envelope.attempt_id,
        )
        sequence = 1
        source: Path | None = None
        try:
            yield self._event(envelope, sequence, ExecutionEventType.STARTED, "preparing", 0)
            sequence += 1
            source = await self._audio.download_to_wav(
                str(envelope.source.url),
                workspace,
                preserve_channels=True,
            )
            yield self._event(envelope, sequence, ExecutionEventType.STAGE, "transcribing", 5)
            sequence += 1
            transcription = await self._transcriber.transcribe_file(
                str(source),
                job_id=envelope.job_id,
                run_id=envelope.run_id,
                track_id=envelope.track_id,
            )
            transcript_text = effective_transcript_text(transcription)
            segments = transcription.get("segments") if isinstance(transcription, dict) else []
            if not isinstance(segments, list):
                segments = []
            edited_transcript = str(envelope.options.get("edited_transcript") or "").strip()
            if edited_transcript and transcript_text:
                transcript_text = self._apply_edited_reference(
                    transcript_text,
                    edited_transcript,
                    transcription,
                )
            yield self._event(
                envelope,
                sequence,
                ExecutionEventType.STAGE,
                "transcribed",
                30,
                {"transcription": transcription},
            )
            sequence += 1
            if not transcript_text:
                moderation = self._no_content_moderation()
                report = self._no_content_report(transcription)
                outcome = ExecutionOutcome(
                    job_id=envelope.job_id,
                    attempt_id=envelope.attempt_id,
                    track_id=envelope.track_id,
                    job_type=envelope.job_type,
                    source_revision=envelope.source.revision,
                    status="completed",
                    result={
                        "job_id": envelope.job_id,
                        "run_id": envelope.run_id,
                        "track_id": envelope.track_id,
                        "job_type": envelope.job_type.value,
                        "transcription": transcription,
                        "moderation": moderation,
                        "categorization": None,
                        "edited_transcript": edited_transcript or None,
                        "report": report,
                        "flagged": False,
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
                return
            yield self._event(envelope, sequence, ExecutionEventType.STAGE, "moderating", 35)
            sequence += 1
            moderation = await self._moderator.moderate(transcript_text)
            yield self._event(
                envelope,
                sequence,
                ExecutionEventType.STAGE,
                "moderated",
                45,
                {"moderation": moderation},
            )
            sequence += 1
            categorization = None
            discovery = None
            content_description = None
            if not moderation.get("flagged"):
                yield self._event(
                    envelope,
                    sequence,
                    ExecutionEventType.STAGE,
                    "categorizing",
                    50,
                )
                sequence += 1
                categorization = await self._categorizer.categorize(
                    transcript=transcript_text,
                    segments=segments,
                    max_tags=int(envelope.options.get("max_tags") or 8),
                    per_track_transcripts={envelope.track_id: transcript_text},
                )
                yield self._event(
                    envelope,
                    sequence,
                    ExecutionEventType.STAGE,
                    "categorized",
                    65,
                    {"categorization": categorization or {}},
                )
                sequence += 1
                yield self._event(
                    envelope,
                    sequence,
                    ExecutionEventType.STAGE,
                    "discovering",
                    68,
                )
                sequence += 1
                profile = await self._discovery.build_profile(
                    transcript_text,
                    content_id=envelope.track_id,
                    track_name=str(envelope.options.get("track_name") or ""),
                    duration_seconds=self._duration(envelope, transcription),
                    source=DiscoverySerialization.coerce_discovery_source(
                        envelope.options.get("source")
                    )
                    or None,
                    speaker=str(envelope.options.get("speaker") or "") or None,
                    categorization=categorization,
                    prior_description=str(
                        envelope.options.get("content_description") or ""
                    )
                    or None,
                )
                discovery, content_description = DiscoverySupport.discovery_result_bundle(
                    profile,
                    duration_seconds=self._duration(envelope, transcription),
                    source=DiscoverySerialization.coerce_discovery_source(
                        envelope.options.get("source")
                    )
                    or None,
                    published_at=str(envelope.options.get("published_at") or "") or None,
                    trending_score=self._optional_float(
                        envelope.options.get("trending_score")
                    ),
                )
                yield self._event(
                    envelope,
                    sequence,
                    ExecutionEventType.STAGE,
                    "discovered",
                    75,
                    {"discovery": discovery or {}},
                )
                sequence += 1
            yield self._event(envelope, sequence, ExecutionEventType.STAGE, "compressing", 80)
            sequence += 1
            compressed_audio, artifact = await self._compress(envelope, source, workspace)
            yield self._event(
                envelope,
                sequence,
                ExecutionEventType.ARTIFACT_PREPARED,
                "compressed",
                95,
                {"compressed_audio": compressed_audio},
            )
            sequence += 1
            result = {
                "job_id": envelope.job_id,
                "run_id": envelope.run_id,
                "job_type": envelope.job_type.value,
                "track_id": envelope.track_id,
                "source_audio_url": str(envelope.source.url),
                "transcription": transcription,
                "moderation": moderation,
                "categorization": categorization,
                "edited_transcript": edited_transcript or None,
                "compressed_audio": compressed_audio,
            }
            if discovery is not None:
                result["discovery"] = discovery
            if content_description:
                result["content_description"] = content_description
            outcome = ExecutionOutcome(
                job_id=envelope.job_id,
                attempt_id=envelope.attempt_id,
                track_id=envelope.track_id,
                job_type=envelope.job_type,
                source_revision=envelope.source.revision,
                status="completed",
                artifacts=(artifact,),
                result=result,
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
            workspace.cleanup()

    async def _compress(
        self,
        envelope: AttemptEnvelope,
        source: Path,
        workspace: AudioWorkspace,
    ):
        bitrate = await self._native.run(
            delivery_bitrate_kbps,
            str(source),
            maximum_kbps=self._maximum_bitrate_kbps,
        )
        mp3_path = Path(
            await convert_wav_file_to_mp3(
                str(source),
                bitrate_kbps=bitrate,
                job_id=envelope.job_id,
                run_id=envelope.run_id,
                track_id=envelope.track_id,
                purpose="pipeline_output",
            )
        )
        source_info = await self._native.run(probe_audio, str(source))
        output_info = await self._native.run(probe_audio, str(mp3_path))
        digest = await self._native.run(self._sha256, mp3_path)
        storage = self._storage_factory.create(envelope.storage)
        key = storage.key("source", f"{envelope.job_id}.mp3")
        artifact = await self._native.run(
            storage.upload_file,
            mp3_path,
            key,
            sha256=digest,
            content_type="audio/mpeg",
        )
        reduction_bytes = source_info["size_bytes"] - output_info["size_bytes"]
        reduction_pct = (
            reduction_bytes / source_info["size_bytes"] * 100
            if source_info["size_bytes"]
            else 0.0
        )
        compressed = {
            "audio_url": artifact.audio_url,
            "b2_key": artifact.object_key,
            "bucket_name": artifact.bucket_name,
            "format": "mp3",
            "bitrate_kbps": bitrate,
            "duration_seconds": round(output_info["duration_seconds"], 3),
            "size_bytes": output_info["size_bytes"],
            "source_size_bytes": source_info["size_bytes"],
            "size_reduction_bytes": reduction_bytes,
            "size_reduction_pct": round(reduction_pct, 3),
        }
        return compressed, artifact

    @staticmethod
    def _apply_edited_reference(
        transcript_text: str,
        edited_ref: str,
        transcription: dict,
    ) -> str:
        def words(value: str) -> set[str]:
            return set(re.sub(r"[^\w\s]", "", value).lower().split())

        heard = words(transcript_text)
        edited = words(edited_ref)
        accuracy = len(heard & edited) / max(len(edited), 1)
        if accuracy < 0.5 and len(edited) >= 3:
            transcription["transcript"] = edited_ref
            transcription["restored"] = True
            transcription["whisper_failed"] = True
            return edited_ref
        restored = restore_punctuation_from_edit(transcript_text, edited_ref)
        corrected = correct_whisper_mishearings(
            restored if restored != transcript_text else transcript_text,
            edited_ref,
        )
        if corrected and corrected != transcript_text:
            transcription["transcript"] = corrected
            transcription["restored"] = True
            return corrected
        return transcript_text

    @staticmethod
    def _no_content_report(transcription: dict | None) -> dict:
        return {
            "flagged": False,
            "code": "content_not_detected",
            "reason": "No usable spoken content was detected in the transcription",
            "transcription": transcription or {},
        }

    @staticmethod
    def _no_content_moderation() -> dict:
        return {
            "flagged": False,
            "severity": "none",
            "intent": "no_content",
            "reason": "No credible speech content was transcribed",
            "flagged_categories": [],
            "blocked_words_found": [],
        }

    @staticmethod
    def _duration(envelope: AttemptEnvelope, transcription: dict) -> float | None:
        value = envelope.options.get("duration_seconds")
        if value is None:
            value = transcription.get("audio_duration") or transcription.get("duration")
        return PipelineWorkflow._optional_float(value)

    @staticmethod
    def _optional_float(value) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest()

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
