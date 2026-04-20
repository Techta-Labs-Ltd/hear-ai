import asyncio
import traceback
from datetime import datetime

import httpx
import sentry_sdk

from app.core.gpu import gpu
from app.core.downloader import download_audio, cleanup_temp
from app.core.recording_fetcher import fetch_recording
from app.core.platform_settings import fetch_platform_settings
from app.models.database import SessionLocal, AiJob
from app.services.callback import callback_service
from app.services.mixer import mixer

MAX_RETRIES = 3
FETCH_RETRIES = 5
FETCH_BASE_DELAY = 3


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
        asyncio.create_task(self._retry_undelivered_callbacks())
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
            if jobs:
                print(f"[WORKER] Recovered {len(jobs)} interrupted jobs")
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
            if not jobs:
                return

            valid_jobs = []
            for job in jobs:
                if job.callback_url and job.callback_url.startswith(("http://", "https://")):
                    valid_jobs.append(job)
                else:
                    print(f"[WORKER] Job {job.id} has invalid callback URL {job.callback_url!r}, marking as delivered")
                    job.callback_delivered = True
            db.commit()

            if not valid_jobs:
                return

            print(f"[WORKER] Retrying {len(valid_jobs)} undelivered callbacks")
            for job in valid_jobs:
                payload = self._build_result_payload(job)
                delivered = await callback_service.send(job.callback_url, payload)
                if delivered:
                    job.callback_delivered = True
                    db.commit()
                    print(f"[WORKER] Delivered callback for job {job.id}")
                else:
                    print(f"[WORKER] Still unable to deliver callback for job {job.id}")
        finally:
            db.close()

    def _build_result_payload(self, job: AiJob) -> dict:
        if job.status == "completed":
            return {
                "job_id": job.id,
                "job_type": job.job_type or "pipeline",
                "status": "completed",
                "result": job.result_json or {},
                "error": None,
            }
        return {
            "job_id": job.id,
            "job_type": job.job_type or "pipeline",
            "status": "failed",
            "result": None,
            "error": job.error or "unknown",
        }

    def _complete_job(self, db, job: AiJob, result: dict) -> None:
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        job.result_json = result
        db.commit()

    async def _loop(self):
        while self._running:
            try:
                job_id = await asyncio.wait_for(self._queue.get(), timeout=5.0)
                await self._process(job_id)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"[WORKER] Loop error: {e}")
                await asyncio.sleep(1)

    async def _fetch_recording_with_retry(self, recording_id: str):
        for attempt in range(FETCH_RETRIES):
            try:
                return await fetch_recording(recording_id)
            except Exception as e:
                if attempt == FETCH_RETRIES - 1:
                    raise
                delay = FETCH_BASE_DELAY * (2 ** attempt)
                print(
                    f"[WORKER] Backend unreachable fetching recording {recording_id}, "
                    f"retry {attempt + 1}/{FETCH_RETRIES} in {delay}s: {e}"
                )
                await asyncio.sleep(delay)

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

            recording = await self._fetch_recording_with_retry(job.recording_id)
            platform = await fetch_platform_settings()
            tracks = recording.tracks
            active_tracks = [t for t in tracks if not t.is_muted]

            print(
                f"[JOB:{job_id[:8]}] START "
                f"type={job.job_type} recording={job.recording_id} "
                f"tracks={len(active_tracks)} attempt={job.attempts}"
            )
            for t in active_tracks:
                print(
                    f"[JOB:{job_id[:8]}]   track={t.track_id[:8]} "
                    f"name={t.name!r} duration={t.duration}s "
                    f"is_enhanced={t.is_enhanced} has_transcription={t.has_transcription}"
                )

            if not active_tracks:
                reason = "no_active_tracks"
                print(f"[JOB:{job_id[:8]}] SKIP — {reason}")
                self._complete_job(
                    db, job,
                    result={
                        "recording_id": job.recording_id,
                        "tracks": {},
                        "master": {},
                        "per_track_transcriptions": {},
                        "skipped": True,
                        "reason": reason,
                        "processing_summary": {
                            "enhanced": 0, "transcribed": 0,
                            "categorized": False, "moderated": False,
                        },
                    },
                )
                if job.callback_url:
                    await callback_service.send(job.callback_url, self._build_result_payload(job))
                db.close()
                return

            track_results = {}
            track_paths = []
            enhanced_track_paths = []

            for track in active_tracks:
                tmp_path = await download_audio(track.audio_url)
                tmp_paths.append(tmp_path)
                track_paths.append({
                    "track_id": track.track_id,
                    "path": tmp_path,
                    "volume": track.volume,
                    "is_muted": track.is_muted,
                    "name": track.name,
                })

                should_enhance = (
                    not job.skip_enhancement
                    and not track.is_enhanced
                    and job.job_type in ("magic_clean", "pipeline")
                )

                if should_enhance:
                    print(f"[JOB:{job_id[:8]}] ENHANCE → track={track.track_id[:8]}")
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
                    print(
                        f"[JOB:{job_id[:8]}] ENHANCE ✓ track={track.track_id[:8]} "
                        f"quality={result.quality_score} snr={result.snr_db}dB "
                        f"url={result.enhanced_url}"
                    )
                    tmp_paths.append(result.local_path)
                    enhanced_track_paths.append({
                        "track_id": track.track_id,
                        "path": result.local_path,
                        "volume": track.volume,
                        "is_muted": False,
                    })
                else:
                    print(f"[JOB:{job_id[:8]}] ENHANCE skip track={track.track_id[:8]} (is_enhanced={track.is_enhanced})")
                    if track.is_enhanced:
                        # Ensure already enhanced tracks are included in mixing and api return datasets
                        enhanced_track_paths.append({
                            "track_id": track.track_id,
                            "path": tmp_path, # Backend provided the enhanced audio directly over audio_url
                            "volume": track.volume,
                            "is_muted": False,
                        })
                        track_results[track.track_id] = {
                            "enhanced_url": track.audio_url,
                            "b2_key": None,
                            "quality_score": track.quality_score if track.quality_score is not None else 1.0,
                            "snr_db": track.snr_db if track.snr_db is not None else 50.0,
                        }

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
            skip_transcription_for_type = job.job_type == "magic_clean"

            if not job.skip_transcription and not skip_transcription_for_type:
                track_id_map = {t.track_id: t for t in active_tracks}
                all_source = enhanced_track_paths or track_paths
                tracks_to_transcribe = [
                    tp for tp in all_source
                    if not track_id_map.get(tp["track_id"], object()).has_transcription
                ]
                for tp in all_source:
                    t = track_id_map.get(tp["track_id"])
                    if t and t.has_transcription:
                        print(f"[JOB:{job_id[:8]}] TRANSCRIBE skip track={tp['track_id'][:8]} (has_transcription=True)")

                if tracks_to_transcribe and not job.existing_transcript:
                    print(f"[JOB:{job_id[:8]}] TRANSCRIBE → {len(tracks_to_transcribe)} track(s)")
                    job.status = "transcribing"
                    db.commit()

                    for tp in tracks_to_transcribe:
                        with open(tp["path"], "rb") as f:
                            audio_bytes = f.read()
                        t_result = await self._transcriber.transcribe(audio_bytes)

                        if t_result.get("silent"):
                            print(
                                f"[JOB:{job_id[:8]}] TRANSCRIBE silent track={tp['track_id'][:8]} "
                                f"— no speech detected, skipping"
                            )
                            continue

                        per_track_transcriptions[tp["track_id"]] = t_result
                        transcript_preview = t_result.get("transcript", "")
                        print(
                            f"[JOB:{job_id[:8]}] TRANSCRIBE ✓ track={tp['track_id'][:8]} "
                            f"lang={t_result.get('language')} "
                            f"words={len(transcript_preview.split())} "
                            f"confidence={t_result.get('confidence')}"
                        )
                        print(f"[JOB:{job_id[:8]}] TRANSCRIPT:\n{transcript_preview}\n")

                    if mixed_path and len(active_tracks) > 1:
                        with open(mixed_path, "rb") as f:
                            mixed_bytes = f.read()
                        combined_transcription = await self._transcriber.transcribe(mixed_bytes)
                        if combined_transcription.get("silent"):
                            combined_transcription = None
                        else:
                            print(
                                f"[JOB:{job_id[:8]}] TRANSCRIBE ✓ master-mix "
                                f"lang={combined_transcription.get('language')} "
                                f"words={len(combined_transcription.get('transcript','').split())}"
                            )
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
                    "silent": False,
                }

            transcript_text = combined_transcription.get("transcript", "").strip() if combined_transcription else ""
            segments = combined_transcription.get("segments", []) if combined_transcription else []

            no_speech = (
                not job.skip_transcription
                and not skip_transcription_for_type
                and not job.existing_transcript
                and not transcript_text
                and not per_track_transcriptions
            )

            if no_speech:
                print(f"[JOB:{job_id[:8]}] NO CONTENT — flagging recording")
                self._complete_job(
                    db, job,
                    result={
                        "recording_id": job.recording_id,
                        "tracks": track_results,
                        "master": master_data,
                        "per_track_transcriptions": {},
                        "skipped": True,
                        "reason": "no_content",
                        "moderation": {
                            "flagged": True,
                            "severity": "high",
                            "intent": "no_content",
                            "reason": "Recording contains no detectable audio content or speech.",
                            "flagged_categories": ["Empty Content"],
                            "blocked_words_found": [],
                        },
                        "processing_summary": {
                            "enhanced": len(track_results),
                            "transcribed": 0,
                            "categorized": False,
                            "moderated": True,
                        },
                    },
                )
                if job.callback_url:
                    await callback_service.send(job.callback_url, self._build_result_payload(job))
                db.close()
                return

            max_tags = job.max_tags if job.max_tags else 8
            categorization_data = None
            if transcript_text and job.job_type in ("tagging", "rebuild", "pipeline"):
                print(f"[JOB:{job_id[:8]}] CATEGORIZE → max_tags={max_tags}")
                job.status = "categorizing"
                db.commit()
                categorization_data = await self._categorizer.categorize(
                    transcript=transcript_text,
                    segments=segments,
                    custom_tags=platform.auto_tag_keywords,
                    max_tags=max_tags,
                )
                print(
                    f"[JOB:{job_id[:8]}] CATEGORIZE ✓ "
                    f"tags={categorization_data.get('tags')} "
                    f"categories={categorization_data.get('categories')} "
                    f"sentiment={categorization_data.get('sentiment')}"
                )

            moderation_data = None
            if transcript_text:
                print(f"[JOB:{job_id[:8]}] MODERATE →")
                job.status = "moderating"
                db.commit()

                combined_mod = await self._moderator.moderate(transcript_text, platform.blocked_keywords)
                track_moderations = {}
                worst = combined_mod
                severity_order = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}

                for track_id, t_data in per_track_transcriptions.items():
                    t_text = t_data.get("transcript", "")
                    if not t_text.strip():
                        continue
                    t_mod = await self._moderator.moderate(t_text, platform.blocked_keywords)
                    track_moderations[track_id] = t_mod
                    if severity_order.get(t_mod["severity"], 0) > severity_order.get(worst["severity"], 0):
                        worst = t_mod
                        worst["reason"] = f"Track {track_id}: {t_mod['reason']}"

                moderation_data = worst
                moderation_data["per_track"] = {
                    tid: {"flagged": m["flagged"], "severity": m["severity"], "reason": m["reason"]}
                    for tid, m in track_moderations.items()
                }
                print(
                    f"[JOB:{job_id[:8]}] MODERATE ✓ "
                    f"flagged={moderation_data.get('flagged')} "
                    f"severity={moderation_data.get('severity')} "
                    f"reason={moderation_data.get('reason')!r}"
                )

            result = {
                "recording_id": job.recording_id,
                "tracks": track_results,
                "master": master_data,
                "per_track_transcriptions": per_track_transcriptions,
                "skipped": False,
                "reason": None,
                "processing_summary": {
                    "enhanced": len(track_results),
                    "transcribed": len(per_track_transcriptions),
                    "categorized": categorization_data is not None,
                    "moderated": moderation_data is not None,
                },
            }

            if combined_transcription:
                result["transcription"] = combined_transcription
            if categorization_data:
                result["categorization"] = categorization_data
            if moderation_data:
                result["moderation"] = moderation_data

            self._complete_job(db, job, result=result)

            duration = (job.completed_at - job.started_at).total_seconds() if job.started_at else 0
            print(
                f"[JOB:{job_id[:8]}] DONE "
                f"type={job.job_type} duration={duration:.1f}s "
                f"tracks_enhanced={len(track_results)} "
                f"transcribed={len(per_track_transcriptions)} "
                f"tags={categorization_data.get('tags') if categorization_data else []} "
                f"callback={job.callback_url}"
            )

            if job.callback_url:
                delivered = await callback_service.send(job.callback_url, self._build_result_payload(job))
                job.callback_delivered = delivered
                db.commit()

            db.close()

        except Exception as e:
            sentry_sdk.capture_exception(e)
            print(f"[WORKER] Job {job_id} failed: {e}\n{traceback.format_exc()}")
            try:
                db = SessionLocal()
                job = db.query(AiJob).filter(AiJob.id == job_id).first()
                if job:
                    non_retryable = isinstance(e, (
                        ValueError, TypeError, AttributeError,
                        httpx.HTTPStatusError,
                    ))
                    if not non_retryable and job.attempts < MAX_RETRIES:
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
                            error_payload = {
                                "job_id": job.id,
                                "job_type": job.job_type or "pipeline",
                                "status": "failed",
                                "result": None,
                                "error": str(e)[:500],
                            }
                            delivered = await callback_service.send(job.callback_url, error_payload)
                            job.callback_delivered = delivered
                            db.commit()
                        db.close()
            except Exception:
                pass

        finally:
            for p in tmp_paths:
                cleanup_temp(p)
            await gpu.release()
