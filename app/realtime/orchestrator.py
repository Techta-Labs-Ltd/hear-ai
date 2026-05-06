import time
import traceback
from datetime import datetime

from app.core.downloader import download_audio, cleanup_temp
from app.core.gpu import gpu
from app.core.platform_settings import fetch_platform_settings
from app.core.recording_fetcher import effective_transcript_text, fetch_track
from app.models.database import SessionLocal, AiJob
from app.realtime.broadcaster import manager
from app.services.callback import callback_service


class PipelineOrchestrator:
    def __init__(self, transcriber, enhancer, categorizer, moderator):
        self.transcriber = transcriber
        self.enhancer = enhancer
        self.categorizer = categorizer
        self.moderator = moderator

    def _get_job(self, job_id: str) -> AiJob | None:
        db = SessionLocal()
        try:
            return db.query(AiJob).filter(AiJob.id == job_id).first()
        finally:
            db.close()

    def _update_job(self, job_id: str, **kwargs):
        db = SessionLocal()
        try:
            job = db.query(AiJob).filter(AiJob.id == job_id).first()
            if job:
                for k, v in kwargs.items():
                    setattr(job, k, v)
                db.commit()
        finally:
            db.close()

    async def process_and_stream(self, job_id: str, run_id: str, track_id: str):
        tmp_paths = []

        try:
            self._update_job(job_id, status="running", current_stage="transcribing", started_at=datetime.utcnow(), run_id=run_id, track_id=track_id)

            await manager.broadcast(job_id, {
                "event": "pipeline_started",
                "job_id": job_id,
                "run_id": run_id,
                "track_id": track_id,
                "timestamp": time.time(),
            })

            track = await fetch_track(track_id)
            await manager.broadcast(job_id, {
                "event": "track_fetched",
                "job_id": job_id,
                "run_id": run_id,
                "track_id": track_id,
                "track_name": track.name,
                "timestamp": time.time(),
            })

            existing_tx = effective_transcript_text(track.transcription) if track.transcription else ""
            platform = await fetch_platform_settings()
            tmp_path = None

            async with gpu.exclusive():
                if existing_tx:
                    transcript = {
                        "transcript": existing_tx,
                        "segments": [],
                        "language": "en",
                        "confidence": 1.0,
                    }
                    transcript_text = existing_tx
                    segments = []
                else:
                    tmp_path = await download_audio(track.audio_url, suffix=".wav")
                    tmp_paths.append(tmp_path)
                    with open(tmp_path, "rb") as f:
                        audio_bytes = f.read()
                    transcript = await self.transcriber.transcribe(audio_bytes)
                    transcript_text = (transcript.get("transcript") or "").strip()
                    segments = transcript.get("segments", []) or []

                await manager.broadcast(job_id, {
                    "event": "transcription_complete",
                    "job_id": job_id,
                    "run_id": run_id,
                    "track_id": track_id,
                    "timestamp": time.time(),
                })

                self._update_job(job_id, status="running", current_stage="moderating")
                moderation_data = await self.moderator.moderate(transcript_text, platform.blocked_keywords)
                await manager.broadcast(job_id, {
                    "event": "moderation_complete",
                    "job_id": job_id,
                    "run_id": run_id,
                    "track_id": track_id,
                    "flagged": moderation_data.get("flagged"),
                    "severity": moderation_data.get("severity"),
                    "timestamp": time.time(),
                })

                categorization_data = None
                if transcript_text and not moderation_data.get("flagged"):
                    self._update_job(job_id, status="running", current_stage="categorizing")
                    categorization_data = await self.categorizer.categorize(
                        transcript=transcript_text,
                        segments=segments,
                        per_track_transcripts={track_id: transcript_text},
                    )
                    await manager.broadcast(job_id, {
                        "event": "categorization_complete",
                        "job_id": job_id,
                        "run_id": run_id,
                        "track_id": track_id,
                        "tags": categorization_data.get("tags", []),
                        "categories": categorization_data.get("categories", []),
                        "sentiment": categorization_data.get("sentiment"),
                        "timestamp": time.time(),
                    })

            result = {
                "track_id": track_id,
                "run_id": run_id,
                "transcription": transcript,
                "moderation": moderation_data,
                "categorization": categorization_data,
            }

            self._update_job(
                job_id,
                status="completed",
                current_stage=None,
                result_json=result,
                completed_at=datetime.utcnow(),
            )

            await manager.broadcast(job_id, {
                "event": "pipeline_complete",
                "job_id": job_id,
                "run_id": run_id,
                "track_id": track_id,
                "timestamp": time.time(),
            })

            job = self._get_job(job_id)
            if job and job.callback_url:
                payload = {
                    "job_id": job_id,
                    "run_id": run_id,
                    "track_id": track_id,
                    "job_type": job.job_type or "pipeline",
                    "status": "completed",
                    "result": result,
                    "error": None,
                }
                delivered = await callback_service.send(job.callback_url, payload)
                self._update_job(job_id, callback_delivered=delivered)

        except Exception as e:
            self._update_job(
                job_id,
                status="failed",
                current_stage=None,
                error=str(e)[:500],
                completed_at=datetime.utcnow(),
            )
            await manager.broadcast(job_id, {
                "event": "pipeline_error",
                "job_id": job_id,
                "run_id": run_id,
                "track_id": track_id,
                "message": str(e),
                "detail": traceback.format_exc(),
                "timestamp": time.time(),
            })
            job = self._get_job(job_id)
            if job and job.callback_url:
                payload = {
                    "job_id": job_id,
                    "run_id": run_id,
                    "track_id": track_id,
                    "job_type": job.job_type or "pipeline",
                    "status": "failed",
                    "result": None,
                    "error": str(e)[:500],
                }
                delivered = await callback_service.send(job.callback_url, payload)
                self._update_job(job_id, callback_delivered=delivered)
        finally:
            for p in tmp_paths:
                cleanup_temp(p)
