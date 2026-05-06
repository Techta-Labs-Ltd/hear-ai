import asyncio
import json
import traceback
from datetime import datetime

import httpx
import sentry_sdk
from sqlalchemy.exc import OperationalError

from app.config import settings
from app.core.db_gate import db_write_lock
from app.core.gpu import gpu, ml_job_lock
from app.core.downloader import download_audio, cleanup_temp
from app.core.recording_fetcher import fetch_track
from app.core.platform_settings import fetch_platform_settings
from app.models.database import SessionLocal, AiJob, AiTrackJob
from app.realtime.broadcaster import manager
from app.services.callback import callback_service

MAX_RETRIES = 3
FETCH_RETRIES = 5
FETCH_BASE_DELAY = 3


class PipelineWorker:
    def __init__(self, enhancer, transcriber, categorizer, moderator, synthesizer):
        self._enhancer = enhancer
        self._transcriber = transcriber
        self._categorizer = categorizer
        self._moderator = moderator
        self._synthesizer = synthesizer
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._loop_task = None
        self._inflight: set[asyncio.Task] = set()
        self._global_limit = asyncio.Semaphore(settings.MAX_CONCURRENT_JOBS)
        self._pipeline_limit = asyncio.Semaphore(settings.MAX_CONCURRENT_PIPELINE_JOBS)
        self._magic_clean_limit = asyncio.Semaphore(settings.MAX_CONCURRENT_MAGIC_CLEAN_JOBS)

    async def start(self):
        self._running = True
        print(
            f"[WORKER] single-queue execution; limits jobs={settings.MAX_CONCURRENT_JOBS} "
            f"pipeline={settings.MAX_CONCURRENT_PIPELINE_JOBS} magic_clean={settings.MAX_CONCURRENT_MAGIC_CLEAN_JOBS} "
            f"gpu={settings.MAX_CONCURRENT_GPU_JOBS}"
        )
        self._recover_jobs()
        asyncio.create_task(self._retry_undelivered_callbacks())
        self._loop_task = asyncio.create_task(self._loop())

    async def stop(self):
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
        if self._inflight:
            await asyncio.gather(*self._inflight, return_exceptions=True)

    def enqueue(self, job_id: str, run_id: str | None = None):
        self._queue.put_nowait((job_id, run_id))

    def _recover_jobs(self):
        db = SessionLocal()
        try:
            jobs = (
                db.query(AiJob)
                .filter(
                    AiJob.status.in_(["queued", "running"]),
                    AiJob.attempts < MAX_RETRIES,
                )
                .all()
            )
            for job in jobs:
                job.attempts += 1
                job.status = "queued"
                job.current_stage = None
            db.commit()
            for job in jobs:
                self.enqueue(job.id, job.run_id)
            if jobs:
                print(f"[WORKER] Recovered {len(jobs)} jobs")
        finally:
            db.close()

    async def _retry_undelivered_callbacks(self):
        await asyncio.sleep(10)
        db = SessionLocal()
        try:
            jobs = (
                db.query(AiJob)
                .filter(
                    AiJob.status.in_(["completed", "failed"]),
                    AiJob.callback_url.isnot(None),
                    AiJob.callback_delivered == False,
                )
                .all()
            )
            for job in jobs:
                async with ml_job_lock:
                    payload = self._build_result_payload(job)
                    delivered = await callback_service.send(job.callback_url, payload)
                    if delivered:
                        job.callback_delivered = True
                        await self._commit_with_retry(db)
        finally:
            db.close()

    def _build_result_payload(self, job: AiJob) -> dict:
        if job.status == "completed":
            result_obj = job.result_json
            if isinstance(result_obj, str):
                try:
                    result_obj = json.loads(result_obj)
                except Exception:
                    result_obj = None
            if not isinstance(result_obj, dict) or not result_obj:
                result_obj = {
                    "job_id": job.id,
                    "run_id": job.run_id,
                    "job_type": job.job_type,
                    "track_id": job.track_id,
                    "status": "completed",
                    "result_missing": True,
                }
            return {
                "job_id": job.id,
                "run_id": job.run_id,
                "track_id": job.track_id,
                "job_type": job.job_type,
                "status": "completed",
                "result": result_obj,
                "error": None,
            }
        return {
            "job_id": job.id,
            "run_id": job.run_id,
            "track_id": job.track_id,
            "job_type": job.job_type,
            "status": "failed",
            "result": None,
            "error": job.error or "unknown",
        }

    async def _deliver_job_callback(self, job_id: str) -> None:
        db = SessionLocal()
        try:
            job = db.query(AiJob).filter(AiJob.id == job_id).first()
            if not job or not job.callback_url:
                return
            delivered = await callback_service.send(job.callback_url, self._build_result_payload(job))
            job.callback_delivered = delivered
            await self._commit_with_retry(db)
        finally:
            db.close()

    async def _commit_with_retry(self, db, retries: int = 5):
        for attempt in range(retries):
            try:
                async with db_write_lock:
                    db.commit()
                return
            except OperationalError as exc:
                db.rollback()
                if "database is locked" in str(exc).lower() and attempt < retries - 1:
                    await asyncio.sleep(0.15 * (2 ** attempt))
                    continue
                raise

    async def _loop(self):
        while self._running:
            try:
                job_id, run_id = await asyncio.wait_for(self._queue.get(), timeout=2.0)
                await self._process_with_limits(job_id, run_id)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"[WORKER] Loop error: {e}")
                await asyncio.sleep(1)

    async def _process_with_limits(self, job_id: str, run_id: str | None):
        db = SessionLocal()
        try:
            job = db.query(AiJob).filter(AiJob.id == job_id).first()
            if not job:
                return
            if run_id and job.run_id != run_id:
                return
            actual_run_id = job.run_id
            job_type = job.job_type or "pipeline"
        finally:
            db.close()

        await self._global_limit.acquire()
        gate = None
        try:
            if job_type in ("pipeline", "rebuild"):
                gate = self._pipeline_limit
            elif job_type == "magic_clean":
                gate = self._magic_clean_limit
            if gate:
                await gate.acquire()
            async with ml_job_lock:
                await self._process(job_id, actual_run_id)
        finally:
            if gate:
                gate.release()
            self._global_limit.release()

    async def _fetch_track_with_retry(self, track_id: str):
        for attempt in range(FETCH_RETRIES):
            try:
                return await fetch_track(track_id)
            except Exception:
                if attempt == FETCH_RETRIES - 1:
                    raise
                await asyncio.sleep(FETCH_BASE_DELAY * (2 ** attempt))

    def _get_or_create_track_run(self, db, job: AiJob):
        entry = (
            db.query(AiTrackJob)
            .filter(
                AiTrackJob.job_id == job.id,
                AiTrackJob.run_id == job.run_id,
                AiTrackJob.track_id == (job.track_id or ""),
            )
            .first()
        )
        if entry:
            return entry
        entry = AiTrackJob(
            job_id=job.id,
            run_id=job.run_id,
            track_id=job.track_id or "",
            job_type=job.job_type or "pipeline",
            status="queued",
            current_stage=None,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.add(entry)
        db.flush()
        return entry

    def _coerce_transcript_text(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return ""
            if stripped.startswith("{") or stripped.startswith("["):
                try:
                    parsed = json.loads(stripped)
                    return self._coerce_transcript_text(parsed)
                except Exception:
                    pass
            return stripped
        if isinstance(value, dict):
            for key in ("transcript", "text", "content", "full_text", "value", "result"):
                nested = value.get(key)
                coerced = self._coerce_transcript_text(nested)
                if coerced:
                    return coerced
            return ""
        if isinstance(value, list):
            parts = [self._coerce_transcript_text(v) for v in value]
            parts = [p for p in parts if p]
            return " ".join(parts).strip()
        return str(value).strip()

    def _coerce_segments(self, value) -> list:
        if isinstance(value, list):
            return value
        return []

    def _coerce_reconstruct_payload(self, value) -> tuple[list[dict], bool]:
        if not isinstance(value, dict):
            return [], True
        raw_changes = value.get("changes")
        same_speaker = bool(value.get("same_speaker", True))
        if not isinstance(raw_changes, list):
            return [], same_speaker
        changes: list[dict] = []
        for item in raw_changes:
            if not isinstance(item, dict):
                continue
            try:
                start = float(item.get("segment_start", 0))
                end = float(item.get("segment_end", 0))
            except Exception:
                continue
            text = self._coerce_transcript_text(item.get("new_text", ""))
            if end <= start or not text:
                continue
            changes.append(
                {
                    "segment_start": start,
                    "segment_end": end,
                    "new_text": text,
                }
            )
        return changes, same_speaker

    def _run_is_current(self, db, job_id: str, run_id: str) -> bool:
        current = db.query(AiJob.run_id).filter(AiJob.id == job_id).scalar()
        return current == run_id

    async def _set_stage(self, db, job: AiJob, track_job: AiTrackJob, stage: str) -> bool:
        if not self._run_is_current(db, job.id, job.run_id):
            return False
        now = datetime.utcnow()
        job.status = "running"
        job.current_stage = stage
        if not job.started_at:
            job.started_at = now
        track_job.status = "running"
        track_job.current_stage = stage
        if not track_job.started_at:
            track_job.started_at = now
        track_job.updated_at = now
        await self._commit_with_retry(db)
        await manager.broadcast(job.id, {
            "event": "stage_changed",
            "job_id": job.id,
            "run_id": job.run_id,
            "track_id": track_job.track_id,
            "job_type": job.job_type,
            "status": job.status,
            "current_stage": stage,
        })
        return True

    async def _complete(self, db, job: AiJob, track_job: AiTrackJob, result: dict) -> bool:
        if not self._run_is_current(db, job.id, job.run_id):
            return False
        now = datetime.utcnow()
        job.status = "completed"
        job.current_stage = None
        job.completed_at = now
        job.result_json = result
        track_job.status = "completed"
        track_job.current_stage = None
        track_job.completed_at = now
        track_job.updated_at = now
        track_job.result_json = result
        await self._commit_with_retry(db)
        await manager.broadcast(job.id, {
            "event": "job_completed",
            "job_id": job.id,
            "run_id": job.run_id,
            "track_id": track_job.track_id,
            "job_type": job.job_type,
            "status": "completed",
            "current_stage": None,
        })
        return True

    async def _process_magic_clean(self, job: AiJob, track_job: AiTrackJob, db):
        if not job.track_id:
            raise ValueError("track_id is required for magic_clean")
        track = await self._fetch_track_with_retry(job.track_id)
        enhanced = {}
        if track.is_enhanced:
            enhanced[track.track_id] = {
                "enhanced_url": track.audio_url,
                "quality_score": track.quality_score,
                "snr_db": track.snr_db,
            }
        else:
            local_path = await download_audio(track.audio_url, suffix=".wav")
            try:
                if not await self._set_stage(db, job, track_job, "enhancing"):
                    return
                out = await self._enhancer.enhance(
                    input_path=local_path,
                    track_id=track.track_id,
                    job_id=f"{job.id}-{track.track_id}",
                )
                enhanced[track.track_id] = {
                    "enhanced_url": out.enhanced_url,
                    "b2_key": out.b2_key,
                    "quality_score": out.quality_score,
                    "snr_db": out.snr_db,
                }
            finally:
                cleanup_temp(local_path)
        result = {
            "job_id": job.id,
            "run_id": job.run_id,
            "job_type": "magic_clean",
            "track_id": track.track_id,
            "enhancement": enhanced.get(track.track_id, {}),
        }
        completed = await self._complete(db, job, track_job, result)
        if not completed:
            return

    async def _process(self, job_id: str, run_id: str):
        callback_job_id = None
        use_gpu = False
        db = SessionLocal()
        tmp_path = None
        try:
            await gpu.acquire()
            use_gpu = True
            job = db.query(AiJob).filter(AiJob.id == job_id).first()
            if not job or job.run_id != run_id:
                return
            if job.status == "cancelled":
                return
            track_job = self._get_or_create_track_run(db, job)
            if job.job_type == "magic_clean":
                await self._process_magic_clean(job, track_job, db)
                if job.callback_url and job.status == "completed":
                    callback_job_id = job.id
                return
            if job.job_type == "reconstruct":
                if not await self._set_stage(db, job, track_job, "reconstructing"):
                    return
                changes, same_speaker = self._coerce_reconstruct_payload(job.custom_tags)
                if not changes:
                    raise ValueError("reconstruct requires non-empty segment changes")
                audio_url = self._coerce_transcript_text(job.input_url or "")
                track_id = self._coerce_transcript_text(job.track_id or "")
                if not audio_url:
                    if not track_id:
                        raise ValueError("reconstruct requires audio_url or track_id")
                    track = await self._fetch_track_with_retry(track_id)
                    audio_url = track.audio_url
                if not audio_url:
                    raise ValueError("reconstruct source audio_url not found")
                source_path = await download_audio(audio_url, suffix=".wav")
                try:
                    rebuilt_audio = await self._synthesizer.reconstruct_segments(
                        original_audio_path=source_path,
                        track_id=track_id or "unknown-track",
                        changes=changes,
                        same_speaker=same_speaker,
                    )
                finally:
                    cleanup_temp(source_path)
                result = {
                    "job_id": job.id,
                    "run_id": job.run_id,
                    "job_type": "reconstruct",
                    "track_id": track_id or None,
                    "source_audio_url": audio_url,
                    "same_speaker": same_speaker,
                    "segments_applied": len(changes),
                    "changes": changes,
                    "reconstructed_audio": {
                        "audio_url": rebuilt_audio.audio_url,
                        "b2_key": rebuilt_audio.b2_key,
                        "duration": rebuilt_audio.duration,
                        "audio_format": "mp3",
                    },
                }
                completed = await self._complete(db, job, track_job, result)
                if not completed:
                    return
                if job.callback_url:
                    callback_job_id = job.id
                return

            track_id = job.track_id or ""
            if not track_id:
                raise ValueError("track_id is required")

            track = await self._fetch_track_with_retry(track_id)
            if not track.audio_url:
                raise ValueError(f"Track {track.track_id} has no audio_url")
            platform = await fetch_platform_settings()

            transcript_data = None
            transcript_text = ""
            segments = []

            if job.job_type == "rebuild":
                if not job.edited_transcript:
                    raise ValueError("edited_transcript is required for rebuild")
                if not await self._set_stage(db, job, track_job, "rebuilding_audio"):
                    return
                original_path = await download_audio(track.audio_url, suffix=".wav")
                try:
                    rebuilt_audio = await self._synthesizer.rebuild_track_audio(
                        original_audio_path=original_path,
                        edited_transcript=job.edited_transcript,
                        track_id=track.track_id,
                        job_id=f"{job.id}-{job.run_id}",
                        original_transcript=self._coerce_transcript_text(track.transcription),
                    )
                finally:
                    cleanup_temp(original_path)
                transcript_text = job.edited_transcript.strip()
                transcript_data = {
                    "transcript": transcript_text,
                    "segments": [],
                    "language": "en",
                    "confidence": 1.0,
                    "edited": True,
                }
            elif job.job_type == "categorization":
                transcript_text = self._coerce_transcript_text(job.edited_transcript or track.transcription or "")
                transcript_data = {
                    "transcript": transcript_text,
                    "segments": [],
                    "language": "en",
                    "confidence": 1.0,
                    "edited": bool(job.edited_transcript),
                }
            else:
                if not await self._set_stage(db, job, track_job, "transcribing"):
                    return
                reused_transcript = self._coerce_transcript_text(track.transcription) if track.transcription else ""
                if reused_transcript and job.job_type != "transcription":
                    transcript_data = {
                        "transcript": reused_transcript,
                        "segments": [],
                        "language": "en",
                        "confidence": 1.0,
                    }
                else:
                    tmp_path = await download_audio(track.audio_url, suffix=".wav")
                    with open(tmp_path, "rb") as f:
                        audio_bytes = f.read()
                    transcript_data = await self._transcriber.transcribe(audio_bytes)
                transcript_text = self._coerce_transcript_text((transcript_data or {}).get("transcript", ""))
                segments = self._coerce_segments((transcript_data or {}).get("segments", []))

            track_job.transcript = transcript_text or None
            track_job.updated_at = datetime.utcnow()
            await self._commit_with_retry(db)

            if job.job_type == "transcription":
                result = {
                    "job_id": job.id,
                    "run_id": job.run_id,
                    "job_type": job.job_type,
                    "track_id": track.track_id,
                    "transcription": transcript_data,
                }
                completed = await self._complete(db, job, track_job, result)
                if not completed:
                    return
                if job.callback_url:
                    callback_job_id = job.id
                return

            if not transcript_text:
                if not await self._set_stage(db, job, track_job, "moderating"):
                    return
                moderation = {
                    "flagged": True,
                    "severity": "high",
                    "intent": "no_content",
                    "reason": "No transcription content",
                    "flagged_categories": ["Empty Content"],
                    "blocked_words_found": [],
                }
                track_job.moderation_json = moderation
                track_job.updated_at = datetime.utcnow()
                await self._commit_with_retry(db)
                result = {
                    "job_id": job.id,
                    "run_id": job.run_id,
                    "track_id": track.track_id,
                    "job_type": job.job_type,
                    "transcription": transcript_data,
                    "moderation": moderation,
                    "categorization": None,
                    "edited_transcript": job.edited_transcript,
                }
                completed = await self._complete(db, job, track_job, result)
                if not completed:
                    return
                if job.callback_url:
                    callback_job_id = job.id
                return

            if not await self._set_stage(db, job, track_job, "moderating"):
                return
            moderation = await self._moderator.moderate(transcript_text, platform.blocked_keywords)
            track_job.moderation_json = moderation
            track_job.updated_at = datetime.utcnow()
            await self._commit_with_retry(db)

            categorization = None
            if not moderation.get("flagged"):
                if not await self._set_stage(db, job, track_job, "categorizing"):
                    return
                categorization = await self._categorizer.categorize(
                    transcript=transcript_text,
                    segments=segments,
                    custom_tags=platform.auto_tag_keywords,
                    max_tags=job.max_tags or 8,
                    per_track_transcripts={track.track_id: transcript_text},
                )
                track_job.categorization_json = categorization
                track_job.updated_at = datetime.utcnow()
                await self._commit_with_retry(db)

            result = {
                "job_id": job.id,
                "run_id": job.run_id,
                "job_type": job.job_type,
                "track_id": track.track_id,
                "source_audio_url": track.audio_url,
                "transcription": transcript_data,
                "moderation": moderation,
                "categorization": categorization,
                "edited_transcript": job.edited_transcript,
                "rebuilt_audio": {
                    "audio_url": rebuilt_audio.audio_url,
                    "b2_key": rebuilt_audio.b2_key,
                    "duration": rebuilt_audio.duration,
                } if job.job_type == "rebuild" else None,
            }
            completed = await self._complete(db, job, track_job, result)
            if not completed:
                return

            if job.callback_url:
                callback_job_id = job.id

        except Exception as e:
            sentry_sdk.capture_exception(e)
            print(f"[WORKER] Job {job_id} failed: {e}\n{traceback.format_exc()}")
            try:
                db.rollback()
            except Exception:
                pass
            fail_db = SessionLocal()
            try:
                job = fail_db.query(AiJob).filter(AiJob.id == job_id).first()
                if not job or job.run_id != run_id:
                    return
                track_job = (
                    fail_db.query(AiTrackJob)
                    .filter(
                        AiTrackJob.job_id == job_id,
                        AiTrackJob.run_id == run_id,
                        AiTrackJob.track_id == (job.track_id or ""),
                    )
                    .first()
                )
                non_retryable = isinstance(e, (ValueError, TypeError, AttributeError, httpx.HTTPStatusError))
                if not non_retryable and job.attempts < MAX_RETRIES:
                    job.status = "queued"
                    job.current_stage = None
                    job.attempts += 1
                    if track_job:
                        track_job.status = "queued"
                        track_job.current_stage = None
                        track_job.error = str(e)[:500]
                        track_job.attempts += 1
                        track_job.updated_at = datetime.utcnow()
                    await self._commit_with_retry(fail_db)
                    self.enqueue(job.id, job.run_id)
                    return
                now = datetime.utcnow()
                job.status = "failed"
                job.current_stage = None
                job.error = str(e)[:500]
                job.completed_at = now
                if track_job:
                    track_job.status = "failed"
                    track_job.current_stage = None
                    track_job.error = str(e)[:500]
                    track_job.completed_at = now
                    track_job.updated_at = now
                await self._commit_with_retry(fail_db)
                if job.callback_url:
                    callback_job_id = job.id
                await manager.broadcast(job.id, {
                    "event": "job_failed",
                    "job_id": job.id,
                    "run_id": job.run_id,
                    "track_id": job.track_id,
                    "job_type": job.job_type,
                    "status": "failed",
                    "current_stage": None,
                    "error": job.error,
                })
            finally:
                fail_db.close()
        finally:
            if tmp_path:
                cleanup_temp(tmp_path)
            if use_gpu:
                try:
                    gpu.idle_sync()
                    await gpu.release()
                except Exception:
                    pass
            db.close()
            if callback_job_id:
                try:
                    await self._deliver_job_callback(callback_job_id)
                except Exception as exc:
                    print(f"[WORKER] Callback delivery failed for {callback_job_id}: {exc}")
