import os
import asyncio
import logging
import re
import time
import traceback
from datetime import datetime

import sentry_sdk
from ray import serve
from ray.serve.handle import DeploymentHandle

from hear.config import settings
from hear.core.db_gate import commit_with_retry
from hear.core.downloader import download_audio
from hear.core.audio_utils import (
    convert_wav_file_to_mp3,
    delivery_bitrate_kbps,
    probe_audio,
)
from hear.core.hear_temp import (
    cleanup_job_temp,
    drop_temp_standalone,
)
from hear.core.processing_context import TrackData, effective_transcript_text
from hear.core.platform_settings import fetch_platform_settings
from hear.core.storage import StorageContextError, storage_for_job
from hear.models.database import SessionLocal, AiJob, AiTrackJob
from hear.services.magic_clean.models import DEFAULT_STEM_LEVELS
from hear.models.discovery import coerce_discovery_source
from hear.models.stages import get_label, get_description, get_stage
from hear.services.categorization.discovery import discovery_result_bundle, get_discovery_service
from hear.services.transcription.service import TranscriptionService
from hear.services.moderation.service import ModerationService
from hear.services.categorization.service import CategorizationService
from hear.services.reconstruction.synthesizer import SpeechSynthesizer
from hear.services.model_client import set_model_client, RayModelClient
from hear.services.jobs.scheduler import FairJobScheduler, PendingJob
from hear.services.reconstruction.diff import (
    compute_edit_segments,
    edit_segments_to_changes,
    restore_punctuation_from_edit,
    correct_whisper_mishearings,
)

os.environ["HF_HUB_OFFLINE"] = os.getenv("HF_HUB_OFFLINE", "0")
os.environ["TRANSFORMERS_OFFLINE"] = os.getenv("TRANSFORMERS_OFFLINE", "0")
os.environ["HF_DATASETS_OFFLINE"] = os.getenv("HF_DATASETS_OFFLINE", "0")

logger = logging.getLogger(__name__)
_recon_logger = logging.getLogger("reconstruct")
_recon_logger.setLevel(logging.INFO)
if not _recon_logger.handlers:
    _recon_fh = logging.FileHandler("/workspace/hear-ai/logs/reconstruct.log")
    _recon_fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    _recon_logger.addHandler(_recon_fh)
    _recon_logger.propagate = False

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


def transcription_only_result(job, track, transcript_data: dict | None) -> dict:
    """Build the terminal payload for jobs whose only output is transcription."""
    return {
        "job_id": job.id,
        "run_id": job.run_id,
        "backend_id": job.backend_id,
        "job_type": job.job_type,
        "track_id": track.track_id,
        "transcription": transcript_data or {},
    }


def audio_tag_result(job, track, transcript: str, suggestions: list[str]) -> dict:
    """Build the compact voice-to-tags response exposed to backend clients."""
    return {
        "job_id": job.id,
        "run_id": job.run_id,
        "backend_id": job.backend_id,
        "job_type": job.job_type,
        "track_id": track.track_id,
        "transcription": transcript,
        "suggestions": suggestions[:2],
    }

STAGE_ESTIMATED = {
    "transcribing": 5,
    "moderating": 1,
    "categorizing": 2,
    "discovering": 3,
    "enhancing": 10,
    "separating": 8,
    "mixing": 2,
    "finalizing": 2,
    "reconstructing": 90,
}
@serve.deployment(
    name="orchestrator",
    ray_actor_options={
        "num_gpus": 0.05,
        "num_cpus": 0.5,
    },
    max_ongoing_requests=100,
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
        magic_clean_handle: DeploymentHandle = None,
    ):

        self._transcription_handle = transcription_handle
        self._llm_handle = llm_handle
        self._fish_speech_handle = fish_speech_handle
        self._small_models_handle = small_models_handle
        self._magic_clean_handle = magic_clean_handle
        self._transcriber = TranscriptionService()
        self._categorizer = CategorizationService()
        self._moderator = ModerationService()
        self._synthesizer = SpeechSynthesizer()

        client = RayModelClient({
            "transcription": transcription_handle,
            "llm": llm_handle,
            "fish_speech": fish_speech_handle,
            "small_models": small_models_handle,
        })
        set_model_client(client)

        self._event_queues: dict[str, asyncio.Queue] = {}
        self._job_stages: dict[str, str] = {}
        self._job_start_times: dict[str, float] = {}
        self._stage_times: dict[str, dict[str, float]] = {}
        self._active_count: int = 0
        self._queued_count: int = 0
        self._job_slots = asyncio.Semaphore(settings.ORCHESTRATOR_MAX_CONCURRENT_JOBS)
        self._fair_scheduler = FairJobScheduler(
            max_active=settings.ORCHESTRATOR_MAX_CONCURRENT_JOBS,
            max_active_per_user=settings.ORCHESTRATOR_MAX_CONCURRENT_JOBS_PER_USER,
            type_limits=settings.ORCHESTRATOR_JOB_TYPE_LIMITS,
        )
        self._dispatch_event = asyncio.Event()
        self._job_type_durations: dict[str, list[float]] = {}
        self._scheduled_runs: set[tuple[str, str]] = set()
        self._recovery_started = False
        self._running = True
        self._dispatcher_task = asyncio.create_task(self._dispatch_loop())
        self._recovery_task = asyncio.create_task(self._recovery_loop())

        print(
            "[ORCHESTRATOR] Initialized | max_concurrent_jobs="
            f"{settings.ORCHESTRATOR_MAX_CONCURRENT_JOBS}"
        )

    def __del__(self):
        self._running = False

    def _push_event(self, job_id: str, event: dict):
        queue = self._event_queues.get(job_id)
        if queue:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass
        if event.get("event") in ("job_completed", "job_failed", "job_cancelled"):
            self._event_queues.pop(job_id, None)

    async def subscribe(self, job_id: str):
        # The job may already be terminal by the time a caller subscribes (it
        # finished before the gRPC Subscribe call reached us, or the caller
        # reconnected after the fact) -- in that case there's no live event
        # queue to wait on, so serve the terminal event straight from the DB
        # instead of hanging until the 120s heartbeat timeout.
        if job_id not in self._event_queues:
            db = SessionLocal()
            try:
                job = db.query(AiJob).filter(AiJob.id == job_id).first()
                if job and job.status in TERMINAL_STATUSES:
                    if job.status == "completed":
                        result = job.result_json or {}
                        event = {
                            "event": "job_completed",
                            "job_id": job.id,
                            "run_id": job.run_id,
                            "backend_id": job.backend_id,
                            "track_id": job.track_id,
                            "job_type": job.job_type,
                            "status": "completed",
                            "current_stage": None,
                            "result": result,
                        }
                        if job.job_type == "audio_tag" and isinstance(result, dict):
                            for key in ("tags", "categories", "media_file_id", "type"):
                                if key in result:
                                    event[key] = result[key]
                        yield event
                    elif job.status == "failed":
                        yield {
                            "event": "job_failed",
                            "job_id": job.id,
                            "run_id": job.run_id,
                            "backend_id": job.backend_id,
                            "track_id": job.track_id,
                            "job_type": job.job_type,
                            "status": job.status,
                            "error": job.error or "",
                        }
                    else:
                        yield {
                            "event": "job_cancelled",
                            "job_id": job.id,
                            "run_id": job.run_id,
                            "backend_id": job.backend_id,
                            "track_id": job.track_id,
                            "job_type": job.job_type,
                            "status": "cancelled",
                            "error": job.error or "",
                        }
                    return
            finally:
                db.close()

        queue = self._event_queues.setdefault(job_id, asyncio.Queue(maxsize=256))
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=120)
                yield event
                if event.get("event") in (
                    "job_completed",
                    "job_failed",
                    "job_cancelled",
                ):
                    self._event_queues.pop(job_id, None)
                    break
            except asyncio.TimeoutError:
                yield {"event": "heartbeat", "job_id": job_id}

    async def get_stats(self) -> dict:
        fair = self._fair_scheduler.stats()
        return {
            "queued": fair["queued"],
            "active": self._active_count,
            "total": self._active_count + fair["queued"],
            "active_users": fair["active_users"],
            "queued_users": fair["queued_users"],
            "active_by_type": fair["active_by_type"],
            "oldest_wait_s": 0.0,
            "estimated_wait_s": 0.0,
            "avg_job_duration_s": round(self._estimate_avg_job_duration()),
        }

    async def process(self, job_id: str, run_id: str):
        async with self._job_slots:
            await self._process(job_id, run_id)

    def _pending_job(self, job_id: str, run_id: str) -> PendingJob | None:
        db = SessionLocal()
        try:
            job = (
                db.query(AiJob)
                .filter(AiJob.id == job_id, AiJob.run_id == run_id)
                .first()
            )
            if job is None or job.status != "queued":
                return None
            options = job.job_options if isinstance(job.job_options, dict) else {}
            user_id = str(options.get("user_id") or "").strip()
            if not user_id:
                logger.error("Queued job %s has no user_id; refusing scheduling", job.id)
                return None
            return PendingJob(
                job_id=job.id,
                run_id=job.run_id,
                user_id=user_id,
                job_type=job.job_type or "pipeline",
            )
        finally:
            db.close()

    def _schedule_job(self, job_id: str, run_id: str) -> bool:
        key = (job_id, run_id)
        if key in self._scheduled_runs:
            return False
        pending = self._pending_job(job_id, run_id)
        if pending is None or not self._fair_scheduler.enqueue(pending):
            return False
        self._scheduled_runs.add(key)
        self._event_queues.setdefault(job_id, asyncio.Queue(maxsize=256))
        levels = self._magic_clean_levels_from_pending(pending)
        queued = self._fair_scheduler.queued_count
        queue_details = {
            **levels,
            "queue_position": queued,
            "total_queued": queued,
        }
        self._push_event(job_id, {
            "event": "job_queued",
            "job_id": job_id,
            "run_id": run_id,
            "job_type": pending.job_type,
            "status": "queued",
            "current_stage": "queued",
            "label": "Waiting fairly",
            "description": "Queued using per-user round-robin scheduling",
            "progress_pct": 0,
            "result": queue_details,
            "position": queued,
            "total_queued": queued,
        })
        self._dispatch_event.set()
        return True

    def _magic_clean_levels_from_pending(self, pending: PendingJob) -> dict:
        if pending.job_type != "magic_clean":
            return {}
        db = SessionLocal()
        try:
            job = db.query(AiJob).filter(AiJob.id == pending.job_id).first()
            return self._magic_clean_levels(job) if job else {}
        finally:
            db.close()

    @staticmethod
    def _magic_clean_levels(job: AiJob) -> dict:
        raw_options = getattr(job, "job_options", None)
        options = raw_options if isinstance(raw_options, dict) else {}
        return {
            "speech": (
                options.get("speech")
                if options.get("speech") is not None
                else DEFAULT_STEM_LEVELS.speech
            ),
            "music": (
                options.get("music")
                if options.get("music") is not None
                else DEFAULT_STEM_LEVELS.music
            ),
            "background": (
                options.get("background")
                if options.get("background") is not None
                else DEFAULT_STEM_LEVELS.background
            ),
            "cut_silence": bool(options.get("cut_silence", False)),
        }

    async def _dispatch_loop(self) -> None:
        while self._running:
            await self._dispatch_event.wait()
            self._dispatch_event.clear()
            while self._running:
                pending = self._fair_scheduler.pop_next()
                if pending is None:
                    break
                asyncio.create_task(self._run_scheduled(pending))

    async def _run_scheduled(self, pending: PendingJob) -> None:
        try:
            await self.process(pending.job_id, pending.run_id)
        finally:
            self._fair_scheduler.complete(pending)
            self._scheduled_runs.discard(pending.key)
            self._dispatch_event.set()

    async def enqueue(self, job_id: str, run_id: str) -> dict:
        return {"scheduled": self._schedule_job(job_id, run_id)}

    async def cancel(self, job_id: str) -> bool:
        db = SessionLocal()
        event = None
        try:
            job = db.query(AiJob).filter(AiJob.id == job_id).first()
            if job is None:
                return False
            if job.status not in TERMINAL_STATUSES:
                job.status = "cancelled"
                job.current_stage = None
                job.completed_at = datetime.utcnow()
                await commit_with_retry(db)
                event = {
                    "event": "job_cancelled",
                    "job_id": job.id,
                    "run_id": job.run_id,
                    "track_id": job.track_id,
                    "job_type": job.job_type,
                    "status": "cancelled",
                    "current_stage": None,
                    "error": "",
                }
                self._fair_scheduler.remove(job_id)
                self._scheduled_runs.discard((job.id, job.run_id))
                self._dispatch_event.set()
            return True
        finally:
            db.close()
            if event:
                self._push_event(job_id, event)

    async def recover_jobs(self):
        if self._recovery_started:
            return
        self._recovery_started = True
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
                    self._schedule_job(row[0], row[1])
                print(f"[ORCHESTRATOR] Recovered {len(rows)} jobs from PG")
        finally:
            db.close()

    async def _recovery_loop(self) -> None:
        await self.recover_jobs()
        while self._running:
            await asyncio.sleep(settings.ORCHESTRATOR_RECOVERY_SECONDS)
            db = SessionLocal()
            try:
                rows = (
                    db.query(AiJob.id, AiJob.run_id)
                    .filter(
                        AiJob.status == "queued",
                        AiJob.attempts < settings.JOB_MAX_RETRIES,
                    )
                    .order_by(AiJob.created_at.asc())
                    .all()
                )
                for row in rows:
                    self._schedule_job(row[0], row[1])
            except Exception:
                logger.exception("queued job recovery scan failed")
            finally:
                db.close()

    async def _process(self, job_id: str, run_id: str):
        db = SessionLocal()
        active = False
        tmp_path = None
        failed_sse = None

        try:
            claimed = (
                db.query(AiJob)
                .filter(
                    AiJob.id == job_id,
                    AiJob.run_id == run_id,
                    AiJob.status == "queued",
                )
                .update(
                    {
                        AiJob.status: "running",
                        AiJob.attempts: AiJob.attempts + 1,
                    },
                    synchronize_session=False,
                )
            )
            await commit_with_retry(db)
            if not claimed:
                return
            active = True
            self._active_count += 1
            self._job_start_times[job_id] = time.time()

            job = db.query(AiJob).filter(AiJob.id == job_id).first()
            if not job or job.run_id != run_id or job.status != "running":
                return
            job_type = job.job_type or "pipeline"
            storage_for_job(job)

            track_job = self._get_or_create_track_run(db, job)

            if job_type == "magic_clean":
                await self._process_magic_clean(job, track_job, db)
                return
            elif job_type == "discovery":
                await self._process_discovery(job, track_job, db)
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
                if job.status == "cancelled":
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
                non_retryable = isinstance(
                    e, (ValueError, TypeError, AttributeError, StorageContextError)
                )
                sanitized_error = self._sanitize_error(e)
                failed_stage = job.current_stage or (
                    track_job.current_stage if track_job else None
                )
                failure_report = {
                    "stage": failed_stage,
                    "error": sanitized_error,
                    "attempt": job.attempts,
                    "retryable": not non_retryable,
                }
                if not non_retryable and job.attempts < settings.JOB_MAX_RETRIES:
                    job.status = "queued"
                    job.current_stage = None
                    if track_job:
                        track_job.status = "queued"
                        track_job.current_stage = None
                        track_job.error = sanitized_error
                        track_job.attempts += 1
                        track_job.updated_at = datetime.utcnow()
                    await commit_with_retry(fail_db)
                    self._push_event(job.id, {
                        "event": "job_retrying",
                        "job_id": job.id,
                        "run_id": job.run_id,
                        "track_id": job.track_id,
                        "job_type": job.job_type,
                        "status": "queued",
                        "current_stage": failed_stage,
                        "label": "Stage failed; retrying",
                        "description": sanitized_error,
                        "progress_pct": 0,
                        "result": {"report": failure_report},
                    })
                    asyncio.create_task(
                        self._retry_after(job.id, job.run_id, 15 * job.attempts)
                    )
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
                    "current_stage": failed_stage,
                    "error": job.error,
                    "result": {"report": failure_report},
                }
            finally:
                fail_db.close()
        finally:
            if tmp_path:
                drop_temp_standalone(tmp_path)
            cleanup_job_temp(db, job_id, run_id)
            db.close()
            if failed_sse:
                self._push_event(failed_sse["job_id"], failed_sse)
            if active:
                self._active_count -= 1

    async def _retry_after(self, job_id: str, run_id: str, delay_seconds: int) -> None:
        await asyncio.sleep(min(max(delay_seconds, 1), 60))
        self._schedule_job(job_id, run_id)

    @staticmethod
    def _track_from_job(job: AiJob) -> TrackData:
        """Build processing context exclusively from submitted job data."""
        raw_options = getattr(job, "job_options", None)
        options = raw_options if isinstance(raw_options, dict) else {}
        return TrackData(
            track_id=job.track_id or "",
            audio_url=getattr(job, "input_url", None) or "",
            name="",
            duration=0,
            transcription=None,
            has_transcription=False,
            content_description=None,
            speaker=None,
            source=options.get("source"),
        )

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
        current = (
            db.query(AiJob.run_id, AiJob.status)
            .filter(AiJob.id == job_id)
            .first()
        )
        return bool(current and current.run_id == run_id and current.status != "cancelled")

    def _coerce_transcript_text(self, value) -> str:
        return effective_transcript_text(value)

    def _coerce_segments(self, value) -> list:
        if isinstance(value, list):
            return value
        return []

    async def _set_stage(
        self,
        db,
        job: AiJob,
        track_job: AiTrackJob,
        stage: str,
        details: dict | None = None,
    ) -> bool:
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
            "backend_id": job.backend_id,
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
            "result": details or {},
        })
        self._push_event(job.id, {
            "event": "queue_position",
            "job_id": job.id,
            "position": 0,
            "total_queued": 0,
            "estimated_wait_s": 0,
        })
        return True

    def _push_stage_result(
        self,
        job: AiJob,
        track_job: AiTrackJob,
        stage: str,
        data: dict,
    ) -> None:
        """Publish a completed stage immediately through the gRPC event stream."""
        s = get_stage(job.job_type or "pipeline", stage)
        self._push_event(job.id, {
            "event": "stage_result",
            "job_id": job.id,
            "run_id": job.run_id,
            "track_id": track_job.track_id,
            "job_type": job.job_type,
            "status": "running",
            "current_stage": stage,
            "label": f"{get_label(job.job_type or 'pipeline', stage)} complete",
            "description": get_description(job.job_type or "pipeline", stage),
            "progress_pct": s.progress_end if s else 0,
            "result": {"stage": stage, "data": data},
        })

    @staticmethod
    def _no_content_report(transcript_data: dict | None) -> dict:
        return {
            "flagged": True,
            "code": "content_not_detected",
            "reason": "No usable spoken content was detected in the transcription",
            "transcription": transcript_data or {},
        }

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
            "backend_id": job.backend_id,
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
        if isinstance(error, StorageContextError):
            return error.code
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

    async def _produce_pipeline_mp3(
        self,
        job: AiJob,
        track,
        source_path: str | None,
        db,
    ) -> dict:
        if source_path is None:
            source_path = await download_audio(
                job.input_url or track.audio_url,
                suffix=".source",
                db=db,
                job_id=job.id,
                run_id=job.run_id,
                track_id=track.track_id,
                purpose="pipeline_encode_source",
            )
        bitrate_kbps = delivery_bitrate_kbps(
            source_path, maximum_kbps=settings.PIPELINE_MP3_BITRATE_KBPS
        )
        mp3_path = await convert_wav_file_to_mp3(
            source_path,
            bitrate_kbps=bitrate_kbps,
            job_id=job.id,
            run_id=job.run_id,
            track_id=track.track_id,
            purpose="pipeline_output",
        )
        source_info = probe_audio(source_path)
        output_info = probe_audio(mp3_path)
        storage = storage_for_job(job)
        b2_key = storage.key("source", f"{job.id}.mp3")
        loop = asyncio.get_running_loop()
        url = await loop.run_in_executor(
            None, storage.upload_file, mp3_path, b2_key, "audio/mpeg"
        )
        reduction_bytes = source_info["size_bytes"] - output_info["size_bytes"]
        reduction_pct = (
            reduction_bytes / source_info["size_bytes"] * 100
            if source_info["size_bytes"]
            else 0.0
        )
        return {
            "audio_url": url,
            "b2_key": b2_key,
            "bucket_name": storage.bucket_name,
            "backend_id": job.backend_id,
            "format": "mp3",
            "bitrate_kbps": bitrate_kbps,
            "duration_seconds": round(output_info["duration_seconds"], 3),
            "size_bytes": output_info["size_bytes"],
            "source_size_bytes": source_info["size_bytes"],
            "size_reduction_bytes": reduction_bytes,
            "size_reduction_pct": round(reduction_pct, 3),
        }

    async def _process_discovery(self, job: AiJob, track_job: AiTrackJob, db) -> None:
        track = self._track_from_job(job)
        transcript_text = ""
        transcript_data = {
            "transcript": transcript_text,
            "segments": [],
            "language": "en",
            "confidence": 1.0,
            "reused": True,
        }
        if not transcript_text:
            if not await self._set_stage(db, job, track_job, "transcribing"):
                return
            audio_path = await download_audio(
                job.input_url or track.audio_url,
                suffix=".source",
                db=db,
                job_id=job.id,
                run_id=job.run_id,
                track_id=track.track_id,
                purpose="discovery_source",
            )
            with open(audio_path, "rb") as audio_file:
                transcript_data = await self._transcriber.transcribe(
                    audio_file.read(),
                    job_id=job.id,
                    run_id=job.run_id,
                    track_id=track.track_id,
                )
            transcript_text = self._coerce_transcript_text(
                (transcript_data or {}).get("transcript", "")
            )
            self._push_stage_result(
                job, track_job, "transcribing", transcript_data or {}
            )

        if not transcript_text:
            report = self._no_content_report(transcript_data)
            await self._complete(db, job, track_job, {
                "job_id": job.id,
                "run_id": job.run_id,
                "job_type": job.job_type,
                "track_id": track.track_id,
                "transcription": transcript_data or {},
                "report": report,
                "flagged": True,
            })
            return

        if not await self._set_stage(db, job, track_job, "discovering"):
            return
        discovery, content_description = await self._run_discovery(
            track, transcript_text, None, source=(job.job_options or {}).get("source")
        )
        discovery_data = discovery or {}
        track_job.discovery_json = discovery_data
        await commit_with_retry(db)
        self._push_stage_result(job, track_job, "discovering", discovery_data)
        result = {
            "job_id": job.id,
            "run_id": job.run_id,
            "job_type": job.job_type,
            "track_id": track.track_id,
            "transcription": transcript_data,
            "discovery": discovery_data,
        }
        if content_description:
            result["content_description"] = content_description
        await self._complete(db, job, track_job, result)

    async def _process_pipeline(self, job: AiJob, track_job: AiTrackJob, db):
        track = self._track_from_job(job)
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
            reused_transcript = ""
            if (
                reused_transcript
                and job.job_type != "transcription"
                and not is_regeneration
                and not job.input_url
            ):
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
                    (job.input_url or track.audio_url), suffix=".wav", db=db,
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
        if job.job_type in {"pipeline", "categorization", "rebuild"}:
            self._push_stage_result(
                job, track_job, "transcribing", transcript_data or {}
            )

        if job.job_type == "audio_tag":
            suggestions: list[str] = []
            if transcript_text:
                tag_data = await self._categorizer.categorize(
                    transcript=transcript_text,
                    segments=segments,
                    custom_tags=platform.auto_tag_keywords,
                    max_tags=2,
                    per_track_transcripts={track.track_id: transcript_text},
                )
                if isinstance(tag_data, dict):
                    suggestions = [
                        str(tag).strip()
                        for tag in tag_data.get("tags", [])
                        if str(tag).strip()
                    ][:2]
            result = audio_tag_result(job, track, transcript_text, suggestions)
            completed = await self._complete(db, job, track_job, result)
            if not completed:
                return
            return

        if job.job_type == "transcription":
            result = transcription_only_result(job, track, transcript_data)
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
            "backend_id": job.backend_id,
                "job_type": job.job_type, "transcription": transcript_data,
                "moderation": moderation, "categorization": None,
                "edited_transcript": job.edited_transcript,
                "report": self._no_content_report(transcript_data),
                "flagged": True,
            }
            self._push_stage_result(
                job, track_job, "moderating", result["report"]
            )
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
        self._push_stage_result(job, track_job, "moderating", moderation)

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
            self._push_stage_result(
                job, track_job, "categorizing", categorization or {}
            )

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
                self._push_stage_result(
                    job, track_job, "discovering", discovery_dict or {}
                )

        compressed_audio = None
        if job.job_type == "pipeline":
            if not await self._set_stage(db, job, track_job, "compressing"):
                return
            compressed_audio = await self._produce_pipeline_mp3(
                job, track, tmp_path, db
            )
            self._push_stage_result(
                job, track_job, "compressing", compressed_audio
            )

        result = {
            "job_id": job.id, "run_id": job.run_id, "job_type": job.job_type,
            "backend_id": job.backend_id,
            "track_id": track.track_id, "source_audio_url": track.audio_url,
            "transcription": transcript_data, "moderation": moderation,
            "categorization": categorization, "edited_transcript": job.edited_transcript,
        }
        if discovery_dict is not None:
            result["discovery"] = discovery_dict
        if content_description:
            result["content_description"] = content_description
        if compressed_audio:
            result["compressed_audio"] = compressed_audio

        completed = await self._complete(db, job, track_job, result)
        if not completed:
            return

    async def _process_magic_clean(self, job: AiJob, track_job: AiTrackJob, db):
        if not job.track_id:
            raise ValueError("track_id is required for magic_clean")
        levels = self._magic_clean_levels(job)
        if not await self._set_stage(db, job, track_job, "downloading", levels):
            return
        track = self._track_from_job(job)
        audio_path = await download_audio(
            (job.input_url or track.audio_url), suffix=".wav", db=db,
            job_id=job.id, run_id=job.run_id, track_id=track.track_id,
            purpose="magic_clean", convert_to_wav=True,
        )
        if not await self._set_stage(db, job, track_job, "separating", levels):
            return
        if self._magic_clean_handle is None:
            raise RuntimeError("magic_clean Ray deployment is unavailable")
        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        if not await self._set_stage(db, job, track_job, "enhancing", levels):
            return
        storage = storage_for_job(job)
        enhancement = await self._magic_clean_handle.enhance.remote(
            audio_bytes=audio_bytes,
            track_id=track.track_id,
            job_id=job.id,
            ai_job_id=job.id,
            ai_run_id=job.run_id,
            speech=levels["speech"],
            music=levels["music"],
            background=levels["background"],
            cut_silence=levels["cut_silence"],
            storage_context=storage.context.model_dump(mode="json"),
        )
        if not await self._set_stage(db, job, track_job, "mixing", levels):
            return
        try:
            drop_temp_standalone(audio_path)
        except Exception:
            pass
        if not await self._set_stage(db, job, track_job, "finalizing", levels):
            return
        result = {
            "job_id": job.id, "run_id": job.run_id, "track_id": track.track_id,
            "backend_id": job.backend_id,
            "job_type": job.job_type, "transcription": {}, "moderation": {},
            "categorization": None, "enhanced": True,
            "enhanced_audio": {
                "audio_url": enhancement.get("enhanced_url"),
                "b2_key": enhancement.get("b2_key"),
                "bucket_name": enhancement.get("bucket_name"),
                "backend_id": job.backend_id,
            },
            "quality": {
                "quality_score": enhancement.get("quality_score"),
                "snr_db": enhancement.get("snr_db"),
                "peak_db": enhancement.get("peak_db"),
                "lufs": enhancement.get("lufs"),
                "clipping_detected": enhancement.get("clipping_detected"),
            },
            "stage_times": enhancement.get("stage_times") or {},
        }
        completed = await self._complete(db, job, track_job, result)
        if not completed:
            return

    async def _process_reconstruct(self, job: AiJob, track_job: AiTrackJob, db):
        changes, same_speaker = self._coerce_reconstruct_payload(job.custom_tags or {})
        if not changes:
            raise ValueError("reconstruct requires changes[] with segment_start/segment_end/new_text")
        track = self._track_from_job(job)
        audio_path = await download_audio(
            (job.input_url or track.audio_url), suffix=".wav", db=db,
            job_id=job.id, run_id=job.run_id, track_id=track.track_id,
            purpose="reconstruct_source", convert_to_wav=True,
        )
        if not await self._set_stage(db, job, track_job, "reconstructing"):
            return
        rebuilt = await self._synthesizer.reconstruct_segments(
            original_audio_path=audio_path,
            changes=changes,
            storage=storage_for_job(job),
            same_speaker=same_speaker,
            job_id=job.id, run_id=job.run_id, track_id=track.track_id,
        )
        result = {
            "job_id": job.id, "run_id": job.run_id, "track_id": track.track_id,
            "backend_id": job.backend_id,
            "job_type": job.job_type, "transcription": {}, "moderation": {},
            "categorization": None, "rebuilt_audio": {
                "audio_url": rebuilt.audio_url, "b2_key": rebuilt.b2_key,
                "duration": rebuilt.duration, "bucket_name": rebuilt.bucket_name,
                "backend_id": job.backend_id,
            },
            "segments": [
                {
                    "segment_start": segment.segment_start,
                    "segment_end": segment.segment_end,
                    "b2_key": segment.b2_key,
                    "audio_url": segment.audio_url,
                    "duration": segment.duration,
                    "is_deletion": segment.is_deletion,
                    "bucket_name": segment.bucket_name,
                    "backend_id": job.backend_id,
                }
                for segment in rebuilt.segments
            ],
            "is_regenerated": True,
        }
        completed = await self._complete(db, job, track_job, result)
        if not completed:
            return

    async def _process_edit_transcript(self, job: AiJob, track_job: AiTrackJob, db):
        track = self._track_from_job(job)
        audio_path = await download_audio(
            (job.input_url or track.audio_url), suffix=".wav", db=db,
            job_id=job.id, run_id=job.run_id, track_id=track.track_id,
            purpose="edit_transcript_source", convert_to_wav=True,
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
            original_audio_path=audio_path,
            changes=changes,
            storage=storage_for_job(job),
            same_speaker=True,
            job_id=job.id, run_id=job.run_id, track_id=track.track_id,
        )
        result = {
            "job_id": job.id, "run_id": job.run_id, "track_id": track.track_id,
            "backend_id": job.backend_id,
            "job_type": job.job_type, "transcription": transcript_data,
            "moderation": {}, "categorization": None, "edited_transcript": edited,
            "rebuilt_audio": {
                "audio_url": rebuilt.audio_url, "b2_key": rebuilt.b2_key,
                "duration": rebuilt.duration, "bucket_name": rebuilt.bucket_name,
                "backend_id": job.backend_id,
            },
            "segments": [
                {
                    "segment_start": segment.segment_start,
                    "segment_end": segment.segment_end,
                    "b2_key": segment.b2_key,
                    "audio_url": segment.audio_url,
                    "duration": segment.duration,
                    "is_deletion": segment.is_deletion,
                    "bucket_name": segment.bucket_name,
                    "backend_id": job.backend_id,
                }
                for segment in rebuilt.segments
            ],
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
            is_deletion = bool(item.get("is_deletion")) or not text
            if end < start or (is_deletion and end <= start):
                continue
            original_text = item.get("original_text")
            changes.append({
                "segment_start": start, "segment_end": end,
                "new_text": text, "original_text": original_text,
                "is_deletion": is_deletion,
            })
        return changes, same_speaker
