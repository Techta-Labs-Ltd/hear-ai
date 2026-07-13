import os
os.environ["HF_HUB_OFFLINE"] = os.getenv("HF_HUB_OFFLINE", "0")
os.environ["TRANSFORMERS_OFFLINE"] = os.getenv("TRANSFORMERS_OFFLINE", "0")
os.environ["HF_DATASETS_OFFLINE"] = os.getenv("HF_DATASETS_OFFLINE", "0")

import asyncio
import json
import logging
import re
import time
import traceback
from datetime import datetime

import sentry_sdk
from ray import serve
from ray.serve.handle import DeploymentHandle

from app.config import settings
from app.core.db_gate import commit_with_retry
from app.core.downloader import download_audio
from app.core.hear_temp import (
    cleanup_job_temp,
    drop_temp_standalone,
)
from app.core.recording_fetcher import effective_transcript_text, fetch_track
from app.core.platform_settings import fetch_platform_settings
from app.models.database import SessionLocal, AiJob, AiTrackJob
from app.models.discovery import coerce_discovery_source
from app.models.stages import get_label, get_description, get_stage
from app.services.discovery import discovery_result_bundle, get_discovery_service
from app.services.transcriber import TranscriptionService
from app.services.moderator import ModerationService
from app.services.categorizer import CategorizationService
from app.services.synthesizer import SpeechSynthesizer
from app.services.enhancer import AudioEnhancer
from app.services.magic_clean_adapter import MagicCleanEnhancer
from app.services.triton_client import set_triton_client, RayModelClient
from app.services.diff_engine import (
    compute_edit_segments,
    edit_segments_to_changes,
    restore_punctuation_from_edit,
    correct_whisper_mishearings,
)

logger = logging.getLogger(__name__)
_recon_logger = logging.getLogger("reconstruct")
_recon_logger.setLevel(logging.INFO)
if not _recon_logger.handlers:
    _recon_fh = logging.FileHandler("/workspace/hear-ai/logs/reconstruct.log")
    _recon_fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    _recon_logger.addHandler(_recon_fh)
    _recon_logger.propagate = False

FETCH_RETRIES = 1
FETCH_BASE_DELAY = 2

STAGE_ESTIMATED = {
    "transcribing": 5,
    "moderating": 1,
    "categorizing": 2,
    "discovering": 3,
    "enhancing": 10,
    "reconstructing": 90,
}
import os as _os

ORCHESTRATOR_RUNTIME_ENV = {
    "env_vars": {
        "PYTHONPATH": "/workspace:/workspace/hear-ai:/workspace/resolver",
        "HF_HUB_OFFLINE": "0",
    }
}
@serve.deployment(
    name="orchestrator",
    ray_actor_options={
        "num_gpus": 0.05,
        "num_cpus": 0.5,
        "runtime_env": ORCHESTRATOR_RUNTIME_ENV,
    },
    max_ongoing_requests=3,
    health_check_period_s=10,
    health_check_timeout_s=30,
)
class Orchestrator:
    def __init__(
        self,
        transcription_handle: DeploymentHandle,
        llm_handle: DeploymentHandle,
        fish_speech_handle: DeploymentHandle,
        small_models_handle: DeploymentHandle,
        deepfilternet_handle: DeploymentHandle = None,
        mossformer2_handle: DeploymentHandle = None,
    ):

        self._transcription_handle = transcription_handle
        self._llm_handle = llm_handle
        self._fish_speech_handle = fish_speech_handle
        self._small_models_handle = small_models_handle
        self._deepfilternet_handle = deepfilternet_handle
        self._mossformer2_handle = mossformer2_handle

        self._enhancer = AudioEnhancer()
        self._magic_clean_enhancer = MagicCleanEnhancer(self._enhancer)
        self._transcriber = TranscriptionService()
        self._categorizer = CategorizationService()
        self._moderator = ModerationService()
        self._synthesizer = SpeechSynthesizer()

        client = RayModelClient()
        client._handles = {
            "transcription": transcription_handle,
            "llm": llm_handle,
            "fish_speech": fish_speech_handle,
            "small_models": small_models_handle,
        }
        if deepfilternet_handle:
            client._handles["deepfilternet"] = deepfilternet_handle
        if mossformer2_handle:
            client._handles["mossformer2"] = mossformer2_handle
        set_triton_client(client)

        self._event_queues: dict[str, asyncio.Queue] = {}
        self._job_stages: dict[str, str] = {}
        self._job_start_times: dict[str, float] = {}
        self._stage_times: dict[str, dict[str, float]] = {}
        self._active_count: int = 0
        self._job_type_durations: dict[str, list[float]] = {}
        self._running = True

        print(f"[ORCHESTRATOR] Initialized | max_ongoing=3")

    def __del__(self):
        self._running = False

    def _push_event(self, job_id: str, event: dict):
        queue = self._event_queues.get(job_id)
        if queue:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass
        if event.get("event") in ("job_completed", "job_failed"):
            self._event_queues.pop(job_id, None)

    async def subscribe(self, job_id: str):
        queue = self._event_queues.setdefault(job_id, asyncio.Queue(maxsize=256))
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=120)
                yield event
                if event.get("event") in ("job_completed", "job_failed"):
                    self._event_queues.pop(job_id, None)
                    break
            except asyncio.TimeoutError:
                yield {"event": "heartbeat", "job_id": job_id}

    async def get_stats(self) -> dict:
        return {
            "queued": 0,
            "active": self._active_count,
            "total": self._active_count,
            "oldest_wait_s": 0.0,
            "estimated_wait_s": 0.0,
            "avg_job_duration_s": round(self._estimate_avg_job_duration()),
        }

    async def process(self, job_id: str, run_id: str):
        await self._process(job_id, run_id)

    async def recover_jobs(self):
        db = SessionLocal()
        try:
            rows = (
                db.query(AiJob.id, AiJob.run_id, AiJob.created_at)
                .filter(
                    AiJob.status.in_(["queued", "running"]),
                    AiJob.attempts < settings.JOB_MAX_RETRIES,
                )
                .order_by(AiJob.created_at.asc())
                .all()
            )
            if rows:
                db.query(AiJob).filter(
                    AiJob.id.in_([r[0] for r in rows])
                ).update(
                    {AiJob.status: "queued", AiJob.current_stage: None},
                    synchronize_session=False,
                )
                await commit_with_retry(db)
                for row in rows:
                    asyncio.create_task(self.process(row[0], row[1]))
                print(f"[ORCHESTRATOR] Recovered {len(rows)} jobs from PG")
        finally:
            db.close()

    async def _process(self, job_id: str, run_id: str):
        db = SessionLocal()
        self._active_count += 1
        self._job_start_times[job_id] = time.time()
        tmp_path = None
        failed_sse = None

        try:
            job = db.query(AiJob).filter(AiJob.id == job_id).first()
            if not job or job.run_id != run_id:
                return
            job.status = "running"
            job.attempts += 1
            await commit_with_retry(db)
            job_type = job.job_type or "pipeline"

            track_job = self._get_or_create_track_run(db, job)

            if job_type == "magic_clean":
                await self._process_magic_clean(job, track_job, db)
                return
            elif job_type == "reconstruct":
                await self._process_reconstruct(job, track_job, db)
                return
            elif job_type == "edit_transcript":
                await self._process_edit_transcript(job, track_job, db)
                return
            elif job_type in ("audio_tag", "categorization"):
                await self._process_pipeline(job, track_job, db)
                return
            elif job_type == "transcription":
                await self._process_pipeline(job, track_job, db)
                return
            else:
                await self._process_pipeline(job, track_job, db)
                return

        except Exception as e:
            sentry_sdk.capture_exception(e)
            print(f"[ORCHESTRATOR] Job {job_id} failed: {e}\n{traceback.format_exc()}")
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
                non_retryable = isinstance(e, (ValueError, TypeError, AttributeError))
                sanitized_error = self._sanitize_error(e)
                if not non_retryable and job.attempts < settings.JOB_MAX_RETRIES:
                    job.status = "queued"
                    job.current_stage = None
                    job.attempts += 1
                    if track_job:
                        track_job.status = "queued"
                        track_job.current_stage = None
                        track_job.error = sanitized_error
                        track_job.attempts += 1
                        track_job.updated_at = datetime.utcnow()
                    await commit_with_retry(fail_db)
                    asyncio.create_task(self.process(job.id, job.run_id))
                    return
                now = datetime.utcnow()
                job.status = "failed"
                job.current_stage = None
                job.error = sanitized_error
                job.completed_at = now
                if track_job:
                    track_job.status = "failed"
                    track_job.current_stage = None
                    track_job.error = sanitized_error
                    track_job.completed_at = now
                    track_job.updated_at = now
                await commit_with_retry(fail_db)
                try:
                    cleanup_job_temp(fail_db, job.id, job.run_id)
                    await commit_with_retry(fail_db)
                except Exception as exc:
                    print(f"[TEMP] cleanup_job_temp on failure failed for {job.id}: {exc}")

                failed_sse = {
                    "event": "job_failed",
                    "job_id": job.id,
                    "run_id": job.run_id,
                    "track_id": job.track_id,
                    "job_type": job.job_type,
                    "status": "failed",
                    "current_stage": None,
                    "error": job.error,
                }
            finally:
                fail_db.close()
        finally:
            if tmp_path:
                drop_temp_standalone(tmp_path)
            db.close()
            if failed_sse:
                self._push_event(failed_sse["job_id"], failed_sse)
            self._active_count -= 1
            self._event_queues.pop(job_id, None)

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

    def _run_is_current(self, db, job_id: str, run_id: str) -> bool:
        current = db.query(AiJob.run_id).filter(AiJob.id == job_id).scalar()
        return current == run_id

    def _coerce_transcript_text(self, value) -> str:
        return effective_transcript_text(value)

    def _coerce_segments(self, value) -> list:
        if isinstance(value, list):
            return value
        return []

    async def _set_stage(self, db, job: AiJob, track_job: AiTrackJob, stage: str) -> bool:
        if not self._run_is_current(db, job.id, job.run_id):
            return False
        self._job_stages[job.id] = stage
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
        await commit_with_retry(db)
        label = get_label(job.job_type or "pipeline", stage)
        description = get_description(job.job_type or "pipeline", stage)
        s = get_stage(job.job_type or "pipeline", stage)
        progress_pct = s.progress_mid if s else 0
        start_time = self._job_start_times.get(job.id, time.time())
        elapsed = round(time.time() - start_time, 1)
        stage_timing = self._stage_times.get(job.id, {})
        estimated = max(0, sum(
            v for k, v in STAGE_ESTIMATED.items()
            if k not in stage_timing
        )) if stage_timing else 0
        self._push_event(job.id, {
            "event": "stage_changed",
            "job_id": job.id,
            "run_id": job.run_id,
            "track_id": track_job.track_id,
            "job_type": job.job_type,
            "status": job.status,
            "current_stage": stage,
            "label": label,
            "description": description,
            "progress_pct": progress_pct,
            "elapsed_seconds": elapsed,
            "estimated_remaining": estimated,
            "stage_started_at": datetime.utcnow().isoformat(),
        })
        self._push_event(job.id, {
            "event": "queue_position",
            "job_id": job.id,
            "position": 0,
            "total_queued": 0,
            "estimated_wait_s": 0,
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
        await commit_with_retry(db)
        try:
            cleanup_job_temp(db, job.id, job.run_id)
            await commit_with_retry(db)
        except Exception as exc:
            print(f"[TEMP] cleanup_job_temp on complete failed for {job.id}: {exc}")
        self._track_job_duration(job)
        self._push_event(job.id, self._job_completed_broadcast(job, track_job, result))
        return True

    def _job_completed_broadcast(self, job: AiJob, track_job: AiTrackJob, result: dict) -> dict:
        event = {
            "event": "job_completed",
            "job_id": job.id,
            "run_id": job.run_id,
            "track_id": track_job.track_id,
            "job_type": job.job_type,
            "status": "completed",
            "current_stage": None,
            "result": result,
        }
        if job.job_type == "audio_tag" and isinstance(result, dict):
            for key in ("tags", "categories", "media_file_id", "type"):
                if key in result:
                    event[key] = result[key]
        return event

    def _track_job_duration(self, job: AiJob):
        if not job.started_at:
            return
        dur = (datetime.utcnow() - job.started_at).total_seconds()
        jtype = job.job_type or "unknown"
        self._job_type_durations.setdefault(jtype, []).append(dur)
        if len(self._job_type_durations[jtype]) > 100:
            self._job_type_durations[jtype] = self._job_type_durations[jtype][-100:]

    def _estimate_avg_job_duration(self) -> float:
        all_durs = [d for dlist in self._job_type_durations.values() for d in dlist]
        if not all_durs:
            return 30.0
        return sum(all_durs) / len(all_durs)

    @staticmethod
    def _sanitize_error(error: Exception) -> str:
        msg = str(error)[:200].lower()
        if any(k in msg for k in ("storage cap", "storage capacity", "quota exceeded", "cap exceeded")):
            return "Cloud storage is full. Free up space or increase your storage cap, then try again."
        if isinstance(error, ValueError):
            if any(k in msg for k in ("download", "audio", "file", "empty", "truncat", "mismatch")):
                return "Audio processing failed. Please check the source file and try again."
            return "Invalid request. Please check your input and try again."
        if isinstance(error, RuntimeError):
            return "Processing failed. Please try again later."
        if isinstance(error, TimeoutError) or "timeout" in msg:
            return "Request timed out. Please try with a shorter segment or smaller text."
        return "An unexpected error occurred. Please try again later."

    async def _run_discovery(
        self, track, transcript_text: str, categorization: dict | None,
        *, partial_transcript: bool = False, source: str | None = None,
    ) -> tuple[dict | None, str | None]:
        duration = float(track.duration) if track.duration else None
        track_category = getattr(track, "category", None)
        track_source = coerce_discovery_source(
            source,
            getattr(track, "source", None),
            track_category if isinstance(track_category, str) else None,
        ) or None
        profile = await get_discovery_service().build_profile(
            transcript_text,
            content_id=track.track_id,
            track_name=track.name or "",
            duration_seconds=duration,
            source=track_source,
            speaker=getattr(track, "speaker", None),
            categorization=categorization,
            prior_description=track.content_description,
            partial_transcript=partial_transcript,
        )
        return discovery_result_bundle(
            profile,
            duration_seconds=duration,
            source=track_source,
            published_at=getattr(track, "published_at", None),
            trending_score=getattr(track, "trending_score", None),
        )

    async def _process_pipeline(self, job: AiJob, track_job: AiTrackJob, db):
        job_id = job.id
        track = await self._fetch_track_with_retry(job.track_id)
        platform = await fetch_platform_settings()

        transcript_text = ""
        segments = []
        tmp_path = None

        if job.job_type == "categorization" and job.edited_transcript:
            transcript_text = job.edited_transcript
            transcript_data = {
                "transcript": transcript_text, "segments": segments,
                "language": "en", "confidence": 1.0, "edited": True,
            }
        elif job.job_type in ("audio_tag",):
            if not await self._set_stage(db, job, track_job, "transcribing"):
                return
            tmp_path = await download_audio(
                track.audio_url, suffix=".wav", db=db,
                job_id=job.id, run_id=job.run_id, track_id=track.track_id,
                purpose="pipeline_source",
            )
            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()
            transcript_data = await self._transcriber.transcribe(
                audio_bytes, job_id=job.id, run_id=job.run_id,
                track_id=track.track_id, short_utterance=True,
            )
            transcript_text = self._coerce_transcript_text(
                (transcript_data or {}).get("transcript", "")
            )
            segments = self._coerce_segments((transcript_data or {}).get("segments", []))
        else:
            is_regeneration = (job.attempts == 0 and job.result_json is None)
            reused_transcript = (
                effective_transcript_text(track.transcription) if track.transcription else ""
            )
            if reused_transcript and job.job_type != "transcription" and not is_regeneration:
                transcript_data = {
                    "transcript": reused_transcript, "segments": [],
                    "language": "en", "confidence": 1.0,
                }
                transcript_text = reused_transcript
                segments = []
            else:
                if not await self._set_stage(db, job, track_job, "transcribing"):
                    return
                tmp_path = await download_audio(
                    track.audio_url, suffix=".wav", db=db,
                    job_id=job.id, run_id=job.run_id, track_id=track.track_id,
                    purpose="pipeline_source",
                )
                with open(tmp_path, "rb") as f:
                    audio_bytes = f.read()
                transcript_data = await self._transcriber.transcribe(
                    audio_bytes, job_id=job.id, run_id=job.run_id,
                    track_id=track.track_id,
                )
                transcript_text = self._coerce_transcript_text(
                    (transcript_data or {}).get("transcript", "")
                )
                segments = self._coerce_segments((transcript_data or {}).get("segments", []))
                edited_ref = (job.edited_transcript or "").strip() or reused_transcript
                if edited_ref and transcript_text:
                    def _strip(s):
                        return set(re.sub(r"[^\w\s]", "", s).lower().split())
                    whisper_words = _strip(transcript_text)
                    edit_words = _strip(edited_ref)
                    word_accuracy = len(whisper_words & edit_words) / max(len(edit_words), 1)
                    fallback_threshold = 0.3 if is_regeneration else 0.5
                    if word_accuracy < fallback_threshold and len(edit_words) >= 3:
                        transcript_text = edited_ref
                        if transcript_data and isinstance(transcript_data, dict):
                            transcript_data["transcript"] = edited_ref
                            transcript_data["restored"] = True
                            transcript_data["whisper_failed"] = True
                    else:
                        restored = restore_punctuation_from_edit(transcript_text, edited_ref)
                        corrected = correct_whisper_mishearings(
                            restored if restored != transcript_text else transcript_text, edited_ref,
                        )
                        if corrected and corrected != transcript_text:
                            transcript_text = corrected
                            if transcript_data and isinstance(transcript_data, dict):
                                transcript_data["transcript"] = corrected
                                transcript_data["restored"] = True

        track_job.updated_at = datetime.utcnow()
        await commit_with_retry(db)

        if job.job_type == "transcription":
            result = {
                "job_id": job.id, "run_id": job.run_id, "job_type": job.job_type,
                "track_id": track.track_id, "transcription": transcript_data,
            }
            completed = await self._complete(db, job, track_job, result)
            if not completed:
                return
            return

        if not transcript_text:
            if not await self._set_stage(db, job, track_job, "moderating"):
                return
            moderation = {
                "flagged": True, "severity": "high", "intent": "no_content",
                "reason": "No transcription content", "flagged_categories": ["Empty Content"],
                "blocked_words_found": [],
            }
            track_job.moderation_json = moderation
            track_job.updated_at = datetime.utcnow()
            await commit_with_retry(db)
            result = {
                "job_id": job.id, "run_id": job.run_id, "track_id": track.track_id,
                "job_type": job.job_type, "transcription": transcript_data,
                "moderation": moderation, "categorization": None,
                "edited_transcript": job.edited_transcript,
            }
            completed = await self._complete(db, job, track_job, result)
            if not completed:
                return
            return

        if not await self._set_stage(db, job, track_job, "moderating"):
            return
        stage_start = time.time()
        moderation = await self._moderator.moderate(transcript_text, platform.blocked_keywords)
        track_job.moderation_json = moderation
        track_job.updated_at = datetime.utcnow()
        await commit_with_retry(db)
        self._stage_times[job.id] = self._stage_times.get(job.id, {})
        self._stage_times[job.id]["moderating"] = round(time.time() - stage_start, 3)

        categorization = None
        if not moderation.get("flagged"):
            if not await self._set_stage(db, job, track_job, "categorizing"):
                return
            stage_start = time.time()
            categorization = await self._categorizer.categorize(
                transcript=transcript_text, segments=segments,
                custom_tags=platform.auto_tag_keywords, max_tags=job.max_tags or 8,
                per_track_transcripts={track.track_id: transcript_text},
            )
            track_job.categorization_json = categorization
            track_job.updated_at = datetime.utcnow()
            await commit_with_retry(db)
            self._stage_times[job.id]["categorizing"] = round(time.time() - stage_start, 3)

        discovery_dict = None
        content_description = None
        if (
            transcript_text
            and not moderation.get("flagged")
            and job.job_type in ("pipeline", "categorization", "rebuild")
        ):
            if await self._set_stage(db, job, track_job, "discovering"):
                discovery_dict, content_description = await self._run_discovery(
                    track, transcript_text, categorization,
                )
                track_job.discovery_json = discovery_dict
                track_job.updated_at = datetime.utcnow()
                await commit_with_retry(db)

        result = {
            "job_id": job.id, "run_id": job.run_id, "job_type": job.job_type,
            "track_id": track.track_id, "source_audio_url": track.audio_url,
            "transcription": transcript_data, "moderation": moderation,
            "categorization": categorization, "edited_transcript": job.edited_transcript,
        }
        if discovery_dict is not None:
            result["discovery"] = discovery_dict
        if content_description:
            result["content_description"] = content_description

        completed = await self._complete(db, job, track_job, result)
        if not completed:
            return

    async def _process_magic_clean(self, job: AiJob, track_job: AiTrackJob, db):
        if not job.track_id:
            raise ValueError("track_id is required for magic_clean")
        track = await self._fetch_track_with_retry(job.track_id)
        audio_path = await download_audio(
            track.audio_url, suffix=".wav", db=db,
            job_id=job.id, run_id=job.run_id, track_id=track.track_id,
            purpose="magic_clean",
        )
        if not await self._set_stage(db, job, track_job, "enhancing"):
            return
        await self._magic_clean_enhancer.enhance(
            input_path=audio_path,
            track_id=track.track_id,
            job_id=job.id,
            ai_job_id=job.id,
            ai_run_id=job.run_id,
        )
        try:
            drop_temp_standalone(audio_path)
        except Exception:
            pass
        result = {
            "job_id": job.id, "run_id": job.run_id, "track_id": track.track_id,
            "job_type": job.job_type, "transcription": {}, "moderation": {},
            "categorization": None, "enhanced": True,
        }
        completed = await self._complete(db, job, track_job, result)
        if not completed:
            return

    async def _process_reconstruct(self, job: AiJob, track_job: AiTrackJob, db):
        changes, same_speaker = self._coerce_reconstruct_payload(job.custom_tags or {})
        if not changes:
            raise ValueError("reconstruct requires changes[] with segment_start/segment_end/new_text")
        track = await self._fetch_track_with_retry(job.track_id)
        audio_path = await download_audio(
            (job.input_url or track.audio_url), suffix=".wav", db=db,
            job_id=job.id, run_id=job.run_id, track_id=track.track_id,
            purpose="reconstruct_source",
        )
        if not await self._set_stage(db, job, track_job, "reconstructing"):
            return
        rebuilt = await self._synthesizer.reconstruct_segments(
            audio_path, changes, same_speaker=same_speaker,
            job_id=job.id, run_id=job.run_id, track_id=track.track_id,
            track=track,
        )
        result = {
            "job_id": job.id, "run_id": job.run_id, "track_id": track.track_id,
            "job_type": job.job_type, "transcription": {}, "moderation": {},
            "categorization": None, "rebuilt_audio": {
                "audio_url": rebuilt.audio_url, "b2_key": rebuilt.b2_key,
                "duration": rebuilt.duration,
            },
            "is_regenerated": True,
        }
        completed = await self._complete(db, job, track_job, result)
        if not completed:
            return

    async def _process_edit_transcript(self, job: AiJob, track_job: AiTrackJob, db):
        track = await self._fetch_track_with_retry(job.track_id)
        audio_path = await download_audio(
            (job.input_url or track.audio_url), suffix=".wav", db=db,
            job_id=job.id, run_id=job.run_id, track_id=track.track_id,
            purpose="edit_transcript_source",
        )
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        transcript_data = await self._transcriber.transcribe(
            audio_bytes, job_id=job.id, run_id=job.run_id, track_id=track.track_id,
        )
        original_text = self._coerce_transcript_text(
            (transcript_data or {}).get("transcript", "")
        )
        segments = self._coerce_segments((transcript_data or {}).get("segments", []))
        if not original_text or not segments:
            raise ValueError("edit_transcript requires valid transcription with segments")
        edited = (job.edited_transcript or "").strip() or original_text
        edit_segs = compute_edit_segments(original_text, edited, segments)
        if not edit_segs:
            raise ValueError("Could not detect any edits between original and edited transcript")
        changes = edit_segments_to_changes(edit_segs)
        if not changes:
            raise ValueError("No changes computed for edit_transcript")
        if not await self._set_stage(db, job, track_job, "reconstructing"):
            return
        rebuilt = await self._synthesizer.reconstruct_segments(
            audio_path, changes, same_speaker=True,
            job_id=job.id, run_id=job.run_id, track_id=track.track_id,
            track=track,
        )
        result = {
            "job_id": job.id, "run_id": job.run_id, "track_id": track.track_id,
            "job_type": job.job_type, "transcription": transcript_data,
            "moderation": {}, "categorization": None, "edited_transcript": edited,
            "rebuilt_audio": {
                "audio_url": rebuilt.audio_url, "b2_key": rebuilt.b2_key,
                "duration": rebuilt.duration,
            },
            "is_regenerated": True,
        }
        completed = await self._complete(db, job, track_job, result)
        if not completed:
            return

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
            if end < start or not text:
                continue
            original_text = item.get("original_text")
            changes.append({
                "segment_start": start, "segment_end": end,
                "new_text": text, "original_text": original_text,
            })
        return changes, same_speaker
