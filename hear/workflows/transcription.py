from __future__ import annotations

import uuid

from hear.contracts.events import ExecutionEvent
from hear.contracts.jobs import AttemptEnvelope
from hear.core.downloader import AudioDownloader
from hear.core.hear_temp import TempWorkspace
from hear.inference.qwen_asr import QwenAsrEngine
from hear.services.transcription.service import TranscriptionService


class TranscriptionWorkflow:
    def __init__(self, engine: QwenAsrEngine) -> None:
        self._engine = engine
        self._service = TranscriptionService(engine)

    async def stream(self, request: AttemptEnvelope):
        sequence = 1
        yield self._event(request, sequence, "started", "preparing", 0)
        sequence += 1
        path = await AudioDownloader.download_audio(
            str(request.source.url),
            suffix=".wav",
            job_id=request.job_id,
            run_id=request.run_id,
            track_id=request.track_id,
            purpose="transcription_source",
            convert_to_wav=True,
            preserve_channels=True,
        )
        try:
            yield self._event(request, sequence, "stage", "transcribing", 5)
            sequence += 1
            result = await self._service.transcribe_file(
                path,
                job_id=request.job_id,
                run_id=request.run_id,
                track_id=request.track_id,
                language=str(request.options.get("language") or "en"),
            )
            yield self._event(
                request,
                sequence,
                "outcome",
                "completed",
                100,
                result={"transcription": result},
            )
        finally:
            TempWorkspace.cleanup_job_temp(None, request.job_id, request.run_id)

    @staticmethod
    def _event(
        request: AttemptEnvelope,
        sequence: int,
        event: str,
        stage: str,
        progress_pct: float,
        result: dict | None = None,
    ) -> ExecutionEvent:
        return ExecutionEvent(
            event_id=str(uuid.uuid4()),
            job_id=request.job_id,
            attempt_id=request.attempt_id,
            track_id=request.track_id,
            job_type=request.job_type,
            source_revision=request.source.revision,
            sequence=sequence,
            event=event,
            stage=stage,
            progress_pct=progress_pct,
            result=result,
        )
