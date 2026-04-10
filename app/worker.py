import asyncio
import traceback
from datetime import datetime

import sentry_sdk

from app.core.gpu import gpu
from app.core.downloader import download_audio, cleanup_temp
from app.core.recording_fetcher import fetch_recording
from app.core.platform_settings import fetch_platform_settings
from app.models.database import SessionLocal, AiJob
from app.services.callback import callback_service
from app.services.mixer import mixer

MAX_RETRIES = 3


class PipelineWorker:
    def __init__(self, enhancer, transcriber, categorizer, moderator):
        self._enhancer = enhancer
        self._transcriber = transcriber
        self._categorizer = categorizer
        self._moderator = moderator
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False

    async def start(self):
        self._running = True
        self._recover_jobs()
        asyncio.create_task(self._loop())

    async def stop(self):
        self._running = False

    def enqueue(self, job_id: str):
        self._queue.put_nowait(job_id)

    def _recover_jobs(self):
        db = SessionLocal()
        try:
            jobs = (
                db.query(AiJob)
                .filter(
                    AiJob.status.in_(["pending", "enhancing", "transcribing", "categorizing", "moderating"]),
                    AiJob.attempts < MAX_RETRIES,
                )
                .all()
            )
            for job in jobs:
                job.attempts += 1
                job.status = "pending"
            db.commit()
            for job in jobs:
                self._queue.put_nowait(job.id)
        finally:
            db.close()

    async def _loop(self):
        while self._running:
            try:
                job_id = await asyncio.wait_for(self._queue.get(), timeout=5.0)
                asyncio.create_task(self._process(job_id))
            except asyncio.TimeoutError:
                continue
            except Exception:
                await asyncio.sleep(1)

    async def _process(self, job_id: str):
        await gpu.acquire()
        tmp_paths = []

        try:
            db = SessionLocal()
            job = db.query(AiJob).filter(AiJob.id == job_id).first()
            if not job:
                return

            job.status = "pending"
            job.started_at = datetime.utcnow()
            db.commit()

            recording = await fetch_recording(job.recording_id)
            platform = await fetch_platform_settings()
            tracks = recording.tracks
            active_tracks = [t for t in tracks if not t.is_muted]

            track_results = {}
            track_paths = []
            enhanced_track_paths = []

            for track in active_tracks:
                tmp_path = await download_audio(track.audio_url)
                tmp_paths.append(tmp_path)
                track_entry = {
                    "track_id": track.track_id,
                    "path": tmp_path,
                    "volume": track.volume,
                    "is_muted": track.is_muted,
                    "name": track.name,
                }
                track_paths.append(track_entry)

                if not job.skip_enhancement:
                    job.status = "enhancing"
                    db.commit()
                    result = await self._enhancer.enhance(
                        input_path=tmp_path,
                        recording_id=job.recording_id,
                        job_id=f"{job.id}-{track.track_id}",
                    )
                    track_results[track.track_id] = {
                        "enhanced_url": result.enhanced_url,
                        "b2_key": result.b2_key,
                        "quality_score": result.quality_score,
                        "snr_db": result.snr_db,
                    }
                    enhanced_path = await download_audio(result.enhanced_url)
                    tmp_paths.append(enhanced_path)
                    enhanced_track_paths.append({
                        "track_id": track.track_id,
                        "path": enhanced_path,
                        "volume": track.volume,
                        "is_muted": False,
                    })

            mix_source = enhanced_track_paths if enhanced_track_paths else track_paths
            master_data = {}
            mixed_path = None

            if len(mix_source) > 1:
                mixed_path = mixer.mix(mix_source)
                if mixed_path:
                    tmp_paths.append(mixed_path)
                    master_data = await mixer.mix_and_upload(mix_source, job.recording_id, job.id)
            elif len(mix_source) == 1:
                mixed_path = mix_source[0]["path"]

            per_track_transcriptions = {}
            combined_transcription = None

            if not job.skip_transcription and not job.existing_transcript:
                job.status = "transcribing"
                db.commit()

                for tp in (enhanced_track_paths or track_paths):
                    with open(tp["path"], "rb") as f:
                        audio_bytes = f.read()
                    t_result = await self._transcriber.transcribe(audio_bytes)
                    per_track_transcriptions[tp["track_id"]] = t_result

                if mixed_path and len(active_tracks) > 1:
                    with open(mixed_path, "rb") as f:
                        mixed_bytes = f.read()
                    combined_transcription = await self._transcriber.transcribe(mixed_bytes)
                elif per_track_transcriptions:
                    first_key = list(per_track_transcriptions.keys())[0]
                    combined_transcription = per_track_transcriptions[first_key]

            elif job.existing_transcript:
                combined_transcription = {
                    "transcript": job.existing_transcript,
                    "segments": [],
                    "language": "en",
                    "language_probability": 1.0,
                    "duration": 0,
                    "confidence": 1.0,
                }

            transcript_text = combined_transcription.get("transcript", "") if combined_transcription else ""
            segments = combined_transcription.get("segments", []) if combined_transcription else []

            categorization_data = None
            if transcript_text:
                job.status = "categorizing"
                db.commit()
                custom_tags = list(set((job.custom_tags or []) + platform.auto_tag_keywords))
                categorization_data = await self._categorizer.categorize(
                    transcript=transcript_text,
                    segments=segments,
                    custom_tags=custom_tags,
                )

            moderation_data = None
            if transcript_text:
                job.status = "moderating"
                db.commit()
                moderation_data = await self._moderator.moderate(transcript_text, platform.blocked_keywords)

            result_payload = {
                "job_id": job.id,
                "job_type": "pipeline",
                "status": "completed",
                "result": {
                    "recording_id": job.recording_id,
                    "tracks": track_results,
                    "master": master_data,
                    "per_track_transcriptions": per_track_transcriptions,
                },
            }

            if combined_transcription:
                result_payload["result"]["transcription"] = combined_transcription
            if categorization_data:
                result_payload["result"]["categorization"] = categorization_data
            if moderation_data:
                result_payload["result"]["moderation"] = moderation_data

            job.status = "completed"
            job.completed_at = datetime.utcnow()
            job.result_json = result_payload["result"]
            db.commit()

            if job.callback_url:
                await callback_service.send(job.callback_url, result_payload)

            db.close()

        except Exception as e:
            sentry_sdk.capture_exception(e)
            print(f"[WORKER] Job {job_id} failed: {e}\n{traceback.format_exc()}")
            try:
                db = SessionLocal()
                job = db.query(AiJob).filter(AiJob.id == job_id).first()
                if job:
                    if job.attempts < MAX_RETRIES:
                        job.status = "pending"
                        job.attempts += 1
                        db.commit()
                        db.close()
                        self._queue.put_nowait(job_id)
                    else:
                        job.status = "failed"
                        job.error = str(e)[:500]
                        job.completed_at = datetime.utcnow()
                        db.commit()
                        if job.callback_url:
                            await callback_service.send(job.callback_url, {
                                "job_id": job.id,
                                "job_type": "pipeline",
                                "status": "failed",
                                "error": str(e)[:500],
                            })
                        db.close()
            except Exception:
                pass

        finally:
            for p in tmp_paths:
                cleanup_temp(p)
            await gpu.release()
