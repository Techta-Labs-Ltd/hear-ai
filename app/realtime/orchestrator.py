import time
import traceback
from datetime import datetime

from app.core.downloader import download_audio, cleanup_temp
from app.core.recording_fetcher import fetch_recording
from app.models.database import SessionLocal, AiJob
from app.realtime.broadcaster import manager
from app.services.callback import callback_service


class PipelineOrchestrator:
    def __init__(self, transcriber, enhancer, categorizer):
        self.transcriber = transcriber
        self.enhancer = enhancer
        self.categorizer = categorizer

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

    async def process_and_stream(self, job_id: str, recording_id: str):
        tmp_paths = []

        try:
            self._update_job(job_id, status="pending", started_at=datetime.utcnow())

            await manager.broadcast(job_id, {
                "event": "pipeline_started",
                "job_id": job_id,
                "recording_id": recording_id,
                "timestamp": time.time(),
            })

            recording = await fetch_recording(recording_id)
            active_tracks = [t for t in recording.tracks if not t.is_muted]

            await manager.broadcast(job_id, {
                "event": "recording_fetched",
                "job_id": job_id,
                "recording_id": recording_id,
                "title": recording.title,
                "track_count": len(active_tracks),
                "timestamp": time.time(),
            })

            all_segments = []
            track_results = {}

            for track in active_tracks:
                tmp_path = await download_audio(track.audio_url)
                tmp_paths.append(tmp_path)

                await manager.broadcast(job_id, {
                    "event": "track_started",
                    "job_id": job_id,
                    "track_id": track.track_id,
                    "track_name": track.name,
                    "timestamp": time.time(),
                })

                if not track.is_enhanced:
                    self._update_job(job_id, status="enhancing")
                    await manager.broadcast(job_id, {
                        "event": "enhancement_started",
                        "job_id": job_id,
                        "track_id": track.track_id,
                        "timestamp": time.time(),
                    })
                    try:
                        result = await self.enhancer.enhance(
                            job_id=f"{job_id}-{track.track_id}",
                            input_path=tmp_path,
                            recording_id=recording_id,
                        )
                        track_results[track.track_id] = {
                            "enhanced_url": result.enhanced_url,
                            "quality_score": result.quality_score,
                            "snr_db": result.snr_db,
                        }
                        await manager.broadcast(job_id, {
                            "event": "enhancement_complete",
                            "job_id": job_id,
                            "track_id": track.track_id,
                            "enhanced_url": result.enhanced_url,
                            "quality_score": result.quality_score,
                            "snr_db": result.snr_db,
                            "timestamp": time.time(),
                        })
                        enhanced_path = await download_audio(result.enhanced_url)
                        tmp_paths.append(enhanced_path)
                        tmp_path = enhanced_path
                    except Exception as e:
                        await manager.broadcast(job_id, {
                            "event": "enhancement_error",
                            "job_id": job_id,
                            "track_id": track.track_id,
                            "message": str(e),
                            "timestamp": time.time(),
                        })
                else:
                    await manager.broadcast(job_id, {
                        "event": "enhancement_skipped",
                        "job_id": job_id,
                        "track_id": track.track_id,
                        "timestamp": time.time(),
                    })

                if not track.has_transcription:
                    self._update_job(job_id, status="transcribing")
                    with open(tmp_path, "rb") as f:
                        audio_bytes = f.read()

                    async for chunk in self.transcriber.stream(audio_bytes):
                        if chunk["type"] == "segment":
                            await manager.broadcast(job_id, {
                                "event": "transcript_segment",
                                "job_id": job_id,
                                "track_id": track.track_id,
                                "segment": chunk,
                                "timestamp": time.time(),
                            })
                            all_segments.append(chunk)
                        elif chunk["type"] == "done":
                            await manager.broadcast(job_id, {
                                "event": "transcript_complete",
                                "job_id": job_id,
                                "track_id": track.track_id,
                                "language": chunk.get("language"),
                                "timestamp": time.time(),
                            })
                        elif chunk["type"] == "error":
                            await manager.broadcast(job_id, {
                                "event": "transcript_error",
                                "job_id": job_id,
                                "track_id": track.track_id,
                                "message": chunk["message"],
                                "timestamp": time.time(),
                            })
                else:
                    await manager.broadcast(job_id, {
                        "event": "transcript_skipped",
                        "job_id": job_id,
                        "track_id": track.track_id,
                        "timestamp": time.time(),
                    })

            categorization_data = None
            if all_segments:
                self._update_job(job_id, status="categorizing")
                full_transcript = " ".join(s.get("text", "") for s in all_segments)
                categorization_data = await self.categorizer.categorize(
                    transcript=full_transcript,
                    segments=all_segments,
                )
                await manager.broadcast(job_id, {
                    "event": "categorization_complete",
                    "job_id": job_id,
                    "tags": categorization_data["tags"],
                    "categories": categorization_data["categories"],
                    "sentiment": categorization_data["sentiment"],
                    "timestamp": time.time(),
                })

            result = {
                "recording_id": recording_id,
                "tracks": track_results,
                "categorization": categorization_data,
            }

            self._update_job(
                job_id,
                status="completed",
                result_json=result,
                completed_at=datetime.utcnow(),
            )

            await manager.broadcast(job_id, {
                "event": "pipeline_complete",
                "job_id": job_id,
                "recording_id": recording_id,
                "timestamp": time.time(),
            })

            job = self._get_job(job_id)
            if job and job.callback_url:
                payload = {
                    "job_id": job_id,
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
                error=str(e)[:500],
                completed_at=datetime.utcnow(),
            )
            await manager.broadcast(job_id, {
                "event": "pipeline_error",
                "job_id": job_id,
                "message": str(e),
                "detail": traceback.format_exc(),
                "timestamp": time.time(),
            })
            job = self._get_job(job_id)
            if job and job.callback_url:
                payload = {
                    "job_id": job_id,
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
