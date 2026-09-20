import asyncio
import logging
import math
import os
import re
import time
import traceback
from datetime import UTC, datetime, timedelta
from functools import partial

import sentry_sdk
from ray import serve
from ray.serve.handle import DeploymentHandle

from hear.config import settings
from hear.core.blocking import AsyncCompletion
from hear.core.db_gate import DatabaseCommitter
from hear.core.downloader import AudioDownloader
from hear.core.hear_temp import TempWorkspace
from hear.core.platform_settings import PlatformSettingsProvider
from hear.core.storage import (
    StorageContextError,
    StorageContexts,
    StorageCredentialsExpiredError,
    StorageCredentialsExpiringError,
)
from hear.models.database import AiJob, AiTrackJob, DatabaseRuntime
from hear.models.discovery import DiscoverySerialization
from hear.models.stages import StageCatalog
from hear.services.categorization.discovery import DiscoverySupport
from hear.services.categorization.service import CategorizationService
from hear.services.jobs.scheduler import FairJobScheduler, PendingJob
from hear.services.magic_clean.lineage import (
    MAGIC_CLEAN_DELIVERED_FILE_SHA256_KEY,
    MAGIC_CLEAN_DELIVERED_PCM_SHA256_KEY,
    MAGIC_CLEAN_ENGINE_REVISION_KEY,
    MAGIC_CLEAN_PARENT_JOB_ID_KEY,
    MAGIC_CLEAN_ROOT_URL_KEY,
    MAGIC_CLEAN_SOURCE_FILE_SHA256_KEY,
    MAGIC_CLEAN_SOURCE_PCM_SHA256_KEY,
    MagicCleanLineageError,
    MagicCleanLineageResolver,
)
from hear.services.magic_clean.models import DEFAULT_STEM_LEVELS
from hear.services.model_client import ModelClientRegistry, RayModelClient
from hear.services.moderation.service import ModerationService
from hear.services.reconstruction.synthesizer import SpeechSynthesizer
from hear.services.transcription.service import TranscriptionService
from hear.utils.audio import convert_wav_file_to_mp3, delivery_bitrate_kbps, probe_audio
from hear.utils.processing_context import TrackData, effective_transcript_text
from hear.utils.transcript_diff import (
    compute_edit_segments,
    correct_whisper_mishearings,
    edit_segments_to_changes,
    restore_punctuation_from_edit,
)

os.environ["HF_HUB_OFFLINE"] = os.getenv("HF_HUB_OFFLINE", "0")
os.environ["TRANSFORMERS_OFFLINE"] = os.getenv("TRANSFORMERS_OFFLINE", "0")
os.environ["HF_DATASETS_OFFLINE"] = os.getenv("HF_DATASETS_OFFLINE", "0")
logger = logging.getLogger(__name__)
_recon_logger = logging.getLogger("reconstruct")
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
RECOVERY_INTERRUPTED_ERROR = "service_restarted"
RECONSTRUCTION_LINEAGE_JOB_TYPES = {"reconstruct", "edit_transcript"}
MAX_RECONSTRUCTION_LINEAGE_DEPTH = 32


class OrchestrationResults:
    @staticmethod
    def _reconstruction_rebuilt_url(result_json: object) -> str:
        """Return the exact persisted rebuilt-track URL from a job result."""
        if not isinstance(result_json, dict):
            return ""
        rebuilt = result_json.get("rebuilt_audio")
        if not isinstance(rebuilt, dict):
            return ""
        return str(rebuilt.get("audio_url") or "").strip()

    @staticmethod
    def _reconstruction_segment_urls(result_json: object) -> set[str]:
        """Return exact isolated-segment URLs, whose timestamps are clip-local."""
        if not isinstance(result_json, dict):
            return set()
        segments = result_json.get("segments")
        if not isinstance(segments, list):
            return set()
        urls: set[str] = set()
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            segment_url = str(segment.get("audio_url") or "").strip()
            if segment_url:
                urls.add(segment_url)
        return urls

    @staticmethod
    def _change_intervals(changes: object) -> tuple[tuple[float, float], ...]:
        """Normalize change intervals for safe same-timeline retry comparison."""
        if not isinstance(changes, list):
            return ()
        intervals: list[tuple[float, float]] = []
        for change in changes:
            if not isinstance(change, dict):
                continue
            try:
                raw_start = change.get("segment_start")
                raw_end = change.get("segment_end")
                if raw_start is None or raw_end is None:
                    continue
                start = float(raw_start)
                end = float(raw_end)
            except (TypeError, ValueError):
                continue
            if math.isfinite(start) and math.isfinite(end) and (end > start):
                intervals.append((round(start, 6), round(end, 6)))
        return tuple(sorted(intervals))

    @staticmethod
    def _job_change_intervals(candidate: object) -> tuple[tuple[float, float], ...]:
        custom_tags = getattr(candidate, "custom_tags", None)
        if isinstance(custom_tags, dict):
            intervals = OrchestrationResults._change_intervals(custom_tags.get("changes"))
            if intervals:
                return intervals
        result_json = getattr(candidate, "result_json", None)
        if isinstance(result_json, dict):
            return OrchestrationResults._change_intervals(result_json.get("segments"))
        return ()

    @staticmethod
    def resolve_reconstruction_reference_url(
        input_url: str,
        *,
        backend_id: str,
        track_id: str,
        changes: list[dict],
        jobs: list[object],
        exclude_job_id: str | None = None,
        max_depth: int = MAX_RECONSTRUCTION_LINEAGE_DEPTH,
    ) -> tuple[str, int]:
        """Resolve a same-interval rebuilt retry to its immutable input root."""
        submitted_url = str(input_url or "").strip()
        if not submitted_url:
            return ("", 0)
        requested_intervals = OrchestrationResults._change_intervals(changes)
        parent_by_output: dict[str, tuple[str, tuple[tuple[float, float], ...]] | None] = {}
        segment_outputs: set[str] = set()
        for candidate in jobs:
            if exclude_job_id and getattr(candidate, "id", None) == exclude_job_id:
                continue
            if str(getattr(candidate, "backend_id", "") or "") != backend_id:
                continue
            if str(getattr(candidate, "track_id", "") or "") != track_id:
                continue
            if str(getattr(candidate, "status", "") or "") != "completed":
                continue
            candidate_type = str(getattr(candidate, "job_type", "") or "").replace("-", "_")
            if candidate_type not in RECONSTRUCTION_LINEAGE_JOB_TYPES:
                continue
            result_json = getattr(candidate, "result_json", None)
            segment_outputs.update(OrchestrationResults._reconstruction_segment_urls(result_json))
            output_url = OrchestrationResults._reconstruction_rebuilt_url(result_json)
            if not output_url:
                continue
            options = getattr(candidate, "job_options", None)
            root_hint = ""
            if isinstance(options, dict):
                root_hint = str(options.get("voice_reference_audio_url") or "").strip()
            parent_url = root_hint or str(getattr(candidate, "input_url", "") or "").strip()
            if not parent_url:
                continue
            parent = (parent_url, OrchestrationResults._job_change_intervals(candidate))
            if output_url not in parent_by_output:
                parent_by_output[output_url] = parent
            elif parent_by_output[output_url] != parent:
                parent_by_output[output_url] = None
        current_url = submitted_url
        visited: set[str] = set()
        for depth in range(max_depth):
            if current_url in visited:
                raise ValueError("reconstruction lineage contains a cycle")
            visited.add(current_url)
            if current_url in segment_outputs:
                raise ValueError(
                    "isolated reconstruction segments cannot be chained as track sources"
                )
            if current_url not in parent_by_output:
                return (current_url, depth)
            resolved_parent: tuple[str, tuple[tuple[float, float], ...]] | None = parent_by_output[
                current_url
            ]
            if resolved_parent is None:
                raise ValueError("reconstruction lineage is ambiguous")
            parent_url, parent_intervals = resolved_parent
            if not requested_intervals or requested_intervals != parent_intervals:
                raise ValueError("chained reconstruction must retry the same original intervals")
            current_url = parent_url
        raise ValueError("reconstruction lineage exceeds the safe depth")

    @staticmethod
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

    @staticmethod
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
    ray_actor_options={"num_gpus": 0.05, "num_cpus": 0.5},
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
        self._categorizer = CategorizationService()
        self._moderator = ModerationService()
        client = RayModelClient(
            {
                "transcription": transcription_handle,
                "llm": llm_handle,
                "fish_speech": fish_speech_handle,
                "small_models": small_models_handle,
            }
        )
        ModelClientRegistry.set_model_client(client)
        self._transcriber = TranscriptionService(client)
        self._synthesizer = SpeechSynthesizer(client, self._transcriber)
        self._event_queues: dict[str, set[asyncio.Queue]] = {}
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
        self._run_tasks: dict[tuple[str, str], asyncio.Task] = {}
        self._recovery_started = False
        self._running = True
        self._dispatcher_task = asyncio.create_task(self._dispatch_loop())
        self._recovery_task = asyncio.create_task(self._recovery_loop())
        print(
            f"[ORCHESTRATOR] Initialized | max_concurrent_jobs={settings.ORCHESTRATOR_MAX_CONCURRENT_JOBS}"
        )

    def __del__(self):
        self._running = False

    def _push_event(self, job_id: str, event: dict):
        for queue in tuple(self._event_queues.get(job_id, ())):
            if queue.full():
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait(
                    {
                        "event": "stream_reset",
                        "job_id": job_id,
                        "error": "subscriber_overflow_reconnect_required",
                    }
                )
            else:
                queue.put_nowait(event.copy())

    def _subscription_snapshot(self, job_id: str) -> dict | None:
        db = DatabaseRuntime.SessionLocal()
        try:
            job = db.query(AiJob).filter(AiJob.id == job_id).first()
            if job is None:
                return None
            if (
                job.status == "queued"
                and job.job_type == "magic_clean"
                and (job.error == StorageCredentialsExpiringError.code)
            ):
                return self._magic_clean_storage_refresh_event(
                    job, previous_stage=job.current_stage
                )
            event = {
                "event": f"job_{job.status}" if job.status in TERMINAL_STATUSES else "job_snapshot",
                "job_id": job.id,
                "run_id": job.run_id,
                "backend_id": job.backend_id,
                "track_id": job.track_id,
                "job_type": job.job_type,
                "status": job.status,
                "current_stage": job.current_stage,
                "error": job.error or "",
            }
            if job.status == "completed":
                result = job.result_json or {}
                event["result"] = result
                if job.job_type == "audio_tag" and isinstance(result, dict):
                    for key in ("tags", "categories", "media_file_id", "type"):
                        if key in result:
                            event[key] = result[key]
            return event
        finally:
            db.close()

    async def subscribe(self, job_id: str):
        queue = asyncio.Queue(maxsize=256)
        self._event_queues.setdefault(job_id, set()).add(queue)
        terminal_events = {"job_completed", "job_failed", "job_cancelled"}
        try:
            snapshot = self._subscription_snapshot(job_id)
            if snapshot is not None:
                yield snapshot
                if snapshot.get("event") in terminal_events:
                    return
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=120)
                except TimeoutError:
                    snapshot = self._subscription_snapshot(job_id)
                    if snapshot is not None and snapshot.get("event") in terminal_events:
                        yield snapshot
                        return
                    yield {"event": "heartbeat", "job_id": job_id}
                    continue
                yield event
                if event.get("event") in terminal_events:
                    return
                if event.get("event") == "stream_reset":
                    snapshot = self._subscription_snapshot(job_id)
                    if snapshot is not None:
                        yield snapshot
                    return
        finally:
            subscribers = self._event_queues.get(job_id)
            if subscribers is not None:
                subscribers.discard(queue)
                if not subscribers:
                    self._event_queues.pop(job_id, None)

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

    @staticmethod
    def _execution_lane(job_type: str) -> str:
        if job_type in {"reconstruct", "edit_transcript"}:
            return "reconstruction"
        return job_type

    def _pending_job(self, job_id: str, run_id: str) -> PendingJob | None:
        db = DatabaseRuntime.SessionLocal()
        try:
            job = (
                db.query(AiJob)
                .filter(AiJob.id == job_id, AiJob.run_id == run_id)
                .with_for_update()
                .first()
            )
            if job is None or job.status != "queued":
                return None
            if (job.job_type or "pipeline") == "magic_clean" and (
                not self._queued_magic_clean_storage_is_ready(db, job)
            ):
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
                lane=self._execution_lane(job.job_type or "pipeline"),
            )
        finally:
            db.close()

    def _queued_magic_clean_storage_is_ready(self, db, job: AiJob) -> bool:
        try:
            storage_context = StorageContexts.decrypt_storage_context(
                job.storage_context_encrypted, require_active=False
            )
        except StorageContextError:
            logger.error(
                "Queued Magic Clean job %s has no usable storage context; refusing scheduling",
                job.id,
            )
            self._mark_queued_magic_clean_storage_refresh_required(db, job)
            return False
        if self._magic_clean_storage_context_has_lifetime(storage_context):
            return True
        logger.info(
            "Queued Magic Clean job %s is waiting for refreshed storage credentials", job.id
        )
        self._mark_queued_magic_clean_storage_refresh_required(db, job)
        return False

    def _mark_queued_magic_clean_storage_refresh_required(self, db, job: AiJob) -> None:
        """Persist a non-terminal credential park without consuming an attempt."""
        track_job = (
            db.query(AiTrackJob)
            .filter(
                AiTrackJob.job_id == job.id,
                AiTrackJob.run_id == job.run_id,
                AiTrackJob.track_id == (job.track_id or ""),
                AiTrackJob.status.in_(("queued", "running")),
            )
            .with_for_update()
            .first()
        )
        previous_stage = job.current_stage or (track_job.current_stage if track_job else None)
        changed = job.error != StorageCredentialsExpiringError.code or job.current_stage is not None
        job.current_stage = None
        job.error = StorageCredentialsExpiringError.code
        if track_job is not None:
            changed = changed or (
                track_job.status != "queued"
                or track_job.current_stage is not None
                or track_job.error != StorageCredentialsExpiringError.code
            )
            track_job.status = "queued"
            track_job.current_stage = None
            track_job.error = StorageCredentialsExpiringError.code
            track_job.updated_at = datetime.now(UTC).replace(tzinfo=None)
        if not changed:
            return
        db.commit()
        self._push_event(
            job.id, self._magic_clean_storage_refresh_event(job, previous_stage=previous_stage)
        )

    def _schedule_job(self, job_id: str, run_id: str) -> bool:
        key = (job_id, run_id)
        if key in self._scheduled_runs:
            return False
        pending = self._pending_job(job_id, run_id)
        if pending is None or not self._fair_scheduler.enqueue(pending):
            return False
        self._scheduled_runs.add(key)
        levels = self._magic_clean_levels_from_pending(pending)
        queued = self._fair_scheduler.queued_count
        queue_details = {**levels, "queue_position": queued, "total_queued": queued}
        self._push_event(
            job_id,
            {
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
            },
        )
        self._dispatch_event.set()
        return True

    def _magic_clean_levels_from_pending(self, pending: PendingJob) -> dict:
        if pending.job_type != "magic_clean":
            return {}
        db = DatabaseRuntime.SessionLocal()
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
            "speech": options.get("speech")
            if options.get("speech") is not None
            else DEFAULT_STEM_LEVELS.speech,
            "music": options.get("music")
            if options.get("music") is not None
            else DEFAULT_STEM_LEVELS.music,
            "background": options.get("background")
            if options.get("background") is not None
            else DEFAULT_STEM_LEVELS.background,
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
                self._start_scheduled_run(pending)

    def _start_scheduled_run(self, pending: PendingJob) -> asyncio.Task[None]:
        task = asyncio.create_task(
            self._run_scheduled(pending), name=f"hear-job:{pending.job_id}:{pending.run_id}"
        )
        self._run_tasks[pending.key] = task
        task.add_done_callback(partial(self._finish_scheduled_run, pending))
        return task

    def _finish_scheduled_run(self, pending: PendingJob, task: asyncio.Task[None]) -> None:
        if self._run_tasks.get(pending.key) is not task:
            return
        self._fair_scheduler.complete(pending)
        self._scheduled_runs.discard(pending.key)
        self._run_tasks.pop(pending.key, None)
        self._dispatch_event.set()

    async def _run_scheduled(self, pending: PendingJob) -> None:
        task = asyncio.current_task()
        try:
            if self._pending_job(pending.job_id, pending.run_id) is None:
                return
            await self.process(pending.job_id, pending.run_id)
        finally:
            if task is not None:
                self._finish_scheduled_run(pending, task)

    def _cancel_scheduled_run(self, job_id: str, run_id: str) -> None:
        key = (job_id, run_id)
        self._fair_scheduler.remove(job_id)
        self._scheduled_runs.discard(key)
        task = self._run_tasks.get(key)
        current = asyncio.current_task()
        if task is not None and task is not current and (not task.done()):
            task.cancel()
        self._dispatch_event.set()

    async def enqueue(self, job_id: str, run_id: str) -> dict:
        return {"scheduled": self._schedule_job(job_id, run_id)}

    async def cancel(self, job_id: str) -> bool:
        db = DatabaseRuntime.SessionLocal()
        event = None
        try:
            job = db.query(AiJob).filter(AiJob.id == job_id).first()
            if job is None:
                return False
            if job.status not in TERMINAL_STATUSES:
                cancelled_at_utc = datetime.now(UTC)
                cancelled_at = cancelled_at_utc.replace(tzinfo=None)
                cancel_values: dict[object, object] = {
                    AiJob.status: "cancelled",
                    AiJob.current_stage: None,
                    AiJob.completed_at: cancelled_at,
                }
                refreshed_options = self._magic_clean_cancellation_options(job, cancelled_at_utc)
                if refreshed_options is not None:
                    cancel_values[AiJob.job_options] = refreshed_options
                cancelled = (
                    db.query(AiJob)
                    .filter(AiJob.id == job_id, AiJob.status.notin_(tuple(TERMINAL_STATUSES)))
                    .update(cancel_values, synchronize_session=False)
                )
                if cancelled != 1:
                    db.rollback()
                    return True
                db.query(AiTrackJob).filter(
                    AiTrackJob.job_id == job.id,
                    AiTrackJob.run_id == job.run_id,
                    AiTrackJob.status.notin_(tuple(TERMINAL_STATUSES)),
                ).update(
                    {
                        AiTrackJob.status: "cancelled",
                        AiTrackJob.current_stage: None,
                        AiTrackJob.completed_at: cancelled_at,
                        AiTrackJob.updated_at: cancelled_at,
                    },
                    synchronize_session=False,
                )
                await DatabaseCommitter.commit_with_retry(db)
                db.refresh(job)
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
                self._cancel_scheduled_run(job.id, job.run_id)
            return True
        finally:
            db.close()
            if event:
                self._push_event(job_id, event)

    @staticmethod
    def _magic_clean_cancellation_options(job: AiJob, cancelled_at: datetime) -> dict | None:
        """Refresh cleanup grace before a cancelled writer is signalled."""
        if getattr(job, "job_type", None) != "magic_clean":
            return None
        options = job.job_options if isinstance(job.job_options, dict) else {}
        tombstone = options.get("magic_clean_cleanup_tombstone")
        if not isinstance(tombstone, dict):
            return None
        key = tombstone.get("b2_key")
        tombstone_run_id = str(tombstone.get("run_id") or "").strip()
        current_run_id = str(getattr(job, "run_id", "") or "").strip()
        if (
            not isinstance(key, str)
            or not key.strip()
            or (not tombstone_run_id)
            or (tombstone_run_id != current_run_id)
        ):
            return None
        refreshed_tombstone = {
            **tombstone,
            "cancellation_requested_at": cancelled_at.isoformat(),
            "not_before": (
                cancelled_at + timedelta(seconds=settings.MAGIC_CLEAN_CLEANUP_GRACE_SECONDS)
            ).isoformat(),
        }
        return {**options, "magic_clean_cleanup_tombstone": refreshed_tombstone}

    @staticmethod
    def _magic_clean_recovery_options(
        *,
        job_type: str | None,
        status: str,
        run_id: str,
        job_options: object,
        recovered_at: datetime,
    ) -> dict | None:
        """Delay cleanup while a crashed remote writer may still be stopping."""
        if job_type != "magic_clean" or status != "running":
            return None
        options = job_options if isinstance(job_options, dict) else {}
        tombstone = options.get("magic_clean_cleanup_tombstone")
        if not isinstance(tombstone, dict):
            return None
        key = tombstone.get("b2_key")
        tombstone_run_id = str(tombstone.get("run_id") or "").strip()
        if (
            not isinstance(key, str)
            or not key.strip()
            or (not tombstone_run_id)
            or (tombstone_run_id != run_id)
        ):
            return None
        return {
            **options,
            "magic_clean_cleanup_tombstone": {
                **tombstone,
                "not_before": (
                    recovered_at + timedelta(seconds=settings.MAGIC_CLEAN_CLEANUP_GRACE_SECONDS)
                ).isoformat(),
            },
        }

    async def recover_jobs(self):
        if self._recovery_started:
            return
        self._recovery_started = True
        db = DatabaseRuntime.SessionLocal()
        try:
            interrupted_rows = (
                db.query(AiJob.id, AiJob.run_id, AiJob.status, AiJob.job_type, AiJob.job_options)
                .filter(AiJob.status == "running")
                .with_for_update()
                .all()
            )
            interrupted = 0
            recovered_at = datetime.now(UTC)
            exhausted_at = recovered_at.replace(tzinfo=None)
            for row in interrupted_rows:
                job_values = {
                    AiJob.status: "failed",
                    AiJob.current_stage: None,
                    AiJob.error: RECOVERY_INTERRUPTED_ERROR,
                    AiJob.completed_at: exhausted_at,
                }
                recovery_options = self._magic_clean_recovery_options(
                    job_type=row[3],
                    status=row[2],
                    run_id=row[1],
                    job_options=row[4],
                    recovered_at=recovered_at,
                )
                if recovery_options is not None:
                    job_values[AiJob.job_options] = recovery_options
                transitioned = (
                    db.query(AiJob)
                    .filter(AiJob.id == row[0], AiJob.run_id == row[1], AiJob.status == "running")
                    .update(job_values, synchronize_session=False)
                )
                if transitioned != 1:
                    continue
                interrupted += 1
                db.query(AiTrackJob).filter(
                    AiTrackJob.job_id == row[0],
                    AiTrackJob.run_id == row[1],
                    AiTrackJob.status.in_(("queued", "running")),
                ).update(
                    {
                        AiTrackJob.status: "failed",
                        AiTrackJob.current_stage: None,
                        AiTrackJob.error: RECOVERY_INTERRUPTED_ERROR,
                        AiTrackJob.completed_at: exhausted_at,
                        AiTrackJob.updated_at: exhausted_at,
                    },
                    synchronize_session=False,
                )
            rows = (
                db.query(AiJob.id, AiJob.run_id, AiJob.created_at)
                .filter(AiJob.status == "queued")
                .order_by(AiJob.created_at.asc())
                .with_for_update()
                .all()
            )
            recovered = []
            for row in rows:
                updated = (
                    db.query(AiJob)
                    .filter(AiJob.id == row[0], AiJob.run_id == row[1], AiJob.status == "queued")
                    .update({AiJob.current_stage: None}, synchronize_session=False)
                )
                if updated == 1:
                    recovered.append(row)
            if interrupted or recovered:
                await DatabaseCommitter.commit_with_retry(db)
            for row in recovered:
                self._schedule_job(row[0], row[1])
            if interrupted or recovered:
                print(
                    f"[ORCHESTRATOR] Recovery complete | requeued={len(recovered)} interrupted={interrupted}"
                )
        finally:
            db.close()

    async def _recovery_loop(self) -> None:
        await self.recover_jobs()
        while self._running:
            await asyncio.sleep(settings.ORCHESTRATOR_RECOVERY_SECONDS)
            db = DatabaseRuntime.SessionLocal()
            try:
                rows = (
                    db.query(AiJob.id, AiJob.run_id)
                    .filter(AiJob.status == "queued")
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
        db = DatabaseRuntime.SessionLocal()
        active = False
        tmp_path = None
        failed_sse = None
        job_type = None
        try:
            job = (
                db.query(AiJob)
                .filter(AiJob.id == job_id, AiJob.run_id == run_id, AiJob.status == "queued")
                .with_for_update()
                .first()
            )
            if job is None:
                return
            job_type = job.job_type or "pipeline"
            if job_type == "magic_clean" and (
                not self._queued_magic_clean_storage_is_ready(db, job)
            ):
                return
            job.status = "running"
            job.attempts = int(job.attempts or 0) + 1
            await DatabaseCommitter.commit_with_retry(db)
            active = True
            self._active_count += 1
            self._job_start_times[job_id] = time.time()
            job_storage = StorageContexts.storage_for_job(job)
            if job_type == "magic_clean":
                self._require_magic_clean_storage_lifetime(job_storage)
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
            if job_type == "magic_clean" and isinstance(
                e, (StorageCredentialsExpiredError, StorageCredentialsExpiringError)
            ):
                try:
                    db.rollback()
                except Exception:
                    pass
                try:
                    parked_event = await self._park_magic_clean_for_storage_refresh(job_id, run_id)
                except Exception:
                    logger.exception(
                        "Could not park Magic Clean job %s for credential refresh", job_id
                    )
                else:
                    if parked_event is not None:
                        failed_sse = parked_event
                        return
            sentry_sdk.capture_exception(e)
            print(f"[ORCHESTRATOR] Job {job_id} failed: {e}\n{traceback.format_exc()}")
            try:
                db.rollback()
            except Exception:
                pass
            fail_db = DatabaseRuntime.SessionLocal()
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
                sanitized_error = self._sanitize_error(e)
                failed_stage = job.current_stage or (track_job.current_stage if track_job else None)
                failure_report = {
                    "stage": failed_stage,
                    "error": sanitized_error,
                    "attempt": job.attempts,
                    "retryable": False,
                }
                now = datetime.utcnow()
                transitioned = (
                    fail_db.query(AiJob)
                    .filter(
                        AiJob.id == job.id,
                        AiJob.run_id == run_id,
                        AiJob.status.in_(("queued", "running")),
                    )
                    .update(
                        {
                            AiJob.status: "failed",
                            AiJob.current_stage: None,
                            AiJob.error: sanitized_error,
                            AiJob.completed_at: now,
                        },
                        synchronize_session=False,
                    )
                )
                if transitioned != 1:
                    fail_db.rollback()
                    return
                if track_job:
                    track_transitioned = (
                        fail_db.query(AiTrackJob)
                        .filter(
                            AiTrackJob.id == track_job.id,
                            AiTrackJob.job_id == job.id,
                            AiTrackJob.run_id == run_id,
                            AiTrackJob.status.in_(("queued", "running")),
                        )
                        .update(
                            {
                                AiTrackJob.status: "failed",
                                AiTrackJob.current_stage: None,
                                AiTrackJob.error: sanitized_error,
                                AiTrackJob.completed_at: now,
                                AiTrackJob.updated_at: now,
                            },
                            synchronize_session=False,
                        )
                    )
                    if track_transitioned != 1:
                        fail_db.rollback()
                        return
                await DatabaseCommitter.commit_with_retry(fail_db)
                try:
                    TempWorkspace.cleanup_job_temp(fail_db, job.id, job.run_id)
                    await DatabaseCommitter.commit_with_retry(fail_db)
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
                    "error": sanitized_error,
                    "result": {"report": failure_report},
                }
            finally:
                fail_db.close()
        finally:
            if tmp_path:
                TempWorkspace.drop_temp_standalone(tmp_path)
            TempWorkspace.cleanup_job_temp(db, job_id, run_id)
            db.close()
            if failed_sse:
                self._push_event(failed_sse["job_id"], failed_sse)
            if active:
                self._active_count -= 1

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
        current = db.query(AiJob.run_id, AiJob.status).filter(AiJob.id == job_id).first()
        return bool(current and current.run_id == run_id and (current.status != "cancelled"))

    @staticmethod
    def _magic_clean_storage_refresh_event(job: AiJob, *, previous_stage: str | None) -> dict:
        return {
            "event": "job_queued",
            "job_id": job.id,
            "run_id": job.run_id,
            "track_id": job.track_id,
            "job_type": job.job_type,
            "status": "queued",
            "current_stage": None,
            "label": "Storage credentials must be refreshed",
            "description": "Magic Clean is paused until storage credentials have the required remaining lifetime",
            "progress_pct": 0,
            "error": StorageCredentialsExpiringError.code,
            "result": {
                "report": {
                    "stage": previous_stage,
                    "error": StorageCredentialsExpiringError.code,
                    "retryable": True,
                }
            },
        }

    async def _park_magic_clean_for_storage_refresh(self, job_id: str, run_id: str) -> dict | None:
        park_db = DatabaseRuntime.SessionLocal()
        try:
            job = park_db.query(AiJob).filter(AiJob.id == job_id).first()
            if (
                job is None
                or job.run_id != run_id
                or job.status != "running"
                or (job.job_type != "magic_clean")
            ):
                return None
            track_job = (
                park_db.query(AiTrackJob)
                .filter(
                    AiTrackJob.job_id == job_id,
                    AiTrackJob.run_id == run_id,
                    AiTrackJob.track_id == (job.track_id or ""),
                )
                .first()
            )
            previous_stage = job.current_stage or (track_job.current_stage if track_job else None)
            parked = (
                park_db.query(AiJob)
                .filter(AiJob.id == job_id, AiJob.run_id == run_id, AiJob.status == "running")
                .update(
                    {
                        AiJob.status: "queued",
                        AiJob.current_stage: None,
                        AiJob.error: StorageCredentialsExpiringError.code,
                        AiJob.attempts: max(int(job.attempts or 0) - 1, 0),
                    },
                    synchronize_session=False,
                )
            )
            if parked != 1:
                park_db.rollback()
                return None
            if track_job is not None:
                track_parked = (
                    park_db.query(AiTrackJob)
                    .filter(
                        AiTrackJob.id == track_job.id,
                        AiTrackJob.job_id == job_id,
                        AiTrackJob.run_id == run_id,
                        AiTrackJob.status.in_(("queued", "running")),
                    )
                    .update(
                        {
                            AiTrackJob.status: "queued",
                            AiTrackJob.current_stage: None,
                            AiTrackJob.error: StorageCredentialsExpiringError.code,
                            AiTrackJob.updated_at: datetime.now(UTC).replace(tzinfo=None),
                        },
                        synchronize_session=False,
                    )
                )
                if track_parked != 1:
                    park_db.rollback()
                    return None
            await DatabaseCommitter.commit_with_retry(park_db)
            return self._magic_clean_storage_refresh_event(job, previous_stage=previous_stage)
        except Exception:
            park_db.rollback()
            raise
        finally:
            park_db.close()

    @staticmethod
    def _magic_clean_storage_context_has_lifetime(storage_context) -> bool:
        expires_at = storage_context.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        remaining_seconds = (expires_at - datetime.now(UTC)).total_seconds()
        required_seconds = settings.MAGIC_CLEAN_STORAGE_CREDENTIAL_MIN_TTL_SECONDS
        return (
            math.isfinite(required_seconds)
            and required_seconds > 0
            and (remaining_seconds >= required_seconds)
        )

    @classmethod
    def _require_magic_clean_storage_lifetime(cls, storage) -> None:
        if not cls._magic_clean_storage_context_has_lifetime(storage.context):
            raise StorageCredentialsExpiringError(
                "storage_credentials_expiring_before_magic_clean_cleanup_window"
            )

    def _coerce_transcript_text(self, value) -> str:
        return effective_transcript_text(value)

    def _coerce_segments(self, value) -> list:
        if isinstance(value, list):
            return value
        return []

    async def _set_stage(
        self, db, job: AiJob, track_job: AiTrackJob, stage: str, details: dict | None = None
    ) -> bool:
        now = datetime.utcnow()
        started_at = job.started_at or now
        track_started_at = track_job.started_at or now
        updated = (
            db.query(AiJob)
            .filter(
                AiJob.id == job.id,
                AiJob.run_id == job.run_id,
                AiJob.status.in_(("queued", "running")),
            )
            .update(
                {
                    AiJob.status: "running",
                    AiJob.current_stage: stage,
                    AiJob.started_at: started_at,
                    AiJob.error: None,
                },
                synchronize_session=False,
            )
        )
        if updated != 1:
            db.rollback()
            return False
        track_updated = (
            db.query(AiTrackJob)
            .filter(
                AiTrackJob.id == track_job.id,
                AiTrackJob.job_id == job.id,
                AiTrackJob.run_id == job.run_id,
                AiTrackJob.status.in_(("queued", "running")),
            )
            .update(
                {
                    AiTrackJob.status: "running",
                    AiTrackJob.current_stage: stage,
                    AiTrackJob.started_at: track_started_at,
                    AiTrackJob.updated_at: now,
                    AiTrackJob.error: None,
                },
                synchronize_session=False,
            )
        )
        if track_updated != 1:
            db.rollback()
            return False
        await DatabaseCommitter.commit_with_retry(db)
        db.refresh(job)
        db.refresh(track_job)
        self._job_stages[job.id] = stage
        label = StageCatalog.get_label(job.job_type or "pipeline", stage)
        description = StageCatalog.get_description(job.job_type or "pipeline", stage)
        s = StageCatalog.get_stage(job.job_type or "pipeline", stage)
        progress_pct = s.progress_mid if s else 0
        start_time = self._job_start_times.get(job.id, time.time())
        elapsed = round(time.time() - start_time, 1)
        stage_timing = self._stage_times.get(job.id, {})
        estimated = (
            max(0, sum((v for k, v in STAGE_ESTIMATED.items() if k not in stage_timing)))
            if stage_timing
            else 0
        )
        self._push_event(
            job.id,
            {
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
            },
        )
        self._push_event(
            job.id,
            {
                "event": "queue_position",
                "job_id": job.id,
                "position": 0,
                "total_queued": 0,
                "estimated_wait_s": 0,
            },
        )
        return True

    def _push_stage_result(self, job: AiJob, track_job: AiTrackJob, stage: str, data: dict) -> None:
        """Publish a completed stage immediately through the gRPC event stream."""
        s = StageCatalog.get_stage(job.job_type or "pipeline", stage)
        self._push_event(
            job.id,
            {
                "event": "stage_result",
                "job_id": job.id,
                "run_id": job.run_id,
                "track_id": track_job.track_id,
                "job_type": job.job_type,
                "status": "running",
                "current_stage": stage,
                "label": f"{StageCatalog.get_label(job.job_type or 'pipeline', stage)} complete",
                "description": StageCatalog.get_description(job.job_type or "pipeline", stage),
                "progress_pct": s.progress_end if s else 0,
                "result": {"stage": stage, "data": data},
            },
        )

    @staticmethod
    def _no_content_report(transcript_data: dict | None) -> dict:
        return {
            "flagged": False,
            "code": "content_not_detected",
            "reason": "No usable spoken content was detected in the transcription",
            "transcription": transcript_data or {},
        }

    @staticmethod
    def _no_content_moderation() -> dict:
        """Return a non-harmful result when no credible speech was transcribed."""
        return {
            "flagged": False,
            "severity": "none",
            "intent": "no_content",
            "reason": "No credible speech content was transcribed",
            "flagged_categories": [],
            "blocked_words_found": [],
        }

    async def _complete(self, db, job: AiJob, track_job: AiTrackJob, result: dict) -> bool:
        now = datetime.utcnow()
        completed = (
            db.query(AiJob)
            .filter(
                AiJob.id == job.id,
                AiJob.run_id == job.run_id,
                AiJob.status.in_(("queued", "running")),
            )
            .update(
                {
                    AiJob.status: "completed",
                    AiJob.current_stage: None,
                    AiJob.completed_at: now,
                    AiJob.result_json: result,
                },
                synchronize_session=False,
            )
        )
        if completed != 1:
            db.rollback()
            return False
        track_completed = (
            db.query(AiTrackJob)
            .filter(
                AiTrackJob.id == track_job.id,
                AiTrackJob.job_id == job.id,
                AiTrackJob.run_id == job.run_id,
                AiTrackJob.status.in_(("queued", "running")),
            )
            .update(
                {
                    AiTrackJob.status: "completed",
                    AiTrackJob.current_stage: None,
                    AiTrackJob.completed_at: now,
                    AiTrackJob.updated_at: now,
                    AiTrackJob.result_json: result,
                },
                synchronize_session=False,
            )
        )
        if track_completed != 1:
            db.rollback()
            return False
        await DatabaseCommitter.commit_with_retry(db)
        db.refresh(job)
        db.refresh(track_job)
        try:
            TempWorkspace.cleanup_job_temp(db, job.id, job.run_id)
            await DatabaseCommitter.commit_with_retry(db)
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
        if type(error).__name__ == "DeploymentUnavailableError":
            return "Required AI processing service is unavailable. Please contact operations."
        storage_capacity_errors = (
            "storage cap",
            "storage capacity",
            "quota exceeded",
            "cap exceeded",
        )
        if any(key in msg for key in storage_capacity_errors):
            return (
                "Cloud storage is full. Free up space or increase your storage cap, then try again."
            )
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
        self,
        track,
        transcript_text: str,
        categorization: dict | None,
        *,
        partial_transcript: bool = False,
        source: str | None = None,
    ) -> tuple[dict | None, str | None]:
        duration = float(track.duration) if track.duration else None
        track_category = getattr(track, "category", None)
        track_source = (
            DiscoverySerialization.coerce_discovery_source(
                source,
                getattr(track, "source", None),
                track_category if isinstance(track_category, str) else None,
            )
            or None
        )
        profile = await DiscoverySupport.get_discovery_service().build_profile(
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
        return DiscoverySupport.discovery_result_bundle(
            profile,
            duration_seconds=duration,
            source=track_source,
            published_at=getattr(track, "published_at", None),
            trending_score=getattr(track, "trending_score", None),
        )

    async def _produce_pipeline_mp3(self, job: AiJob, track, source_path: str | None, db) -> dict:
        if source_path is None:
            source_path = await AudioDownloader.download_audio(
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
        storage = StorageContexts.storage_for_job(job)
        b2_key = storage.key("source", f"{job.id}.mp3")
        loop = asyncio.get_running_loop()
        url = await loop.run_in_executor(None, storage.upload_file, mp3_path, b2_key, "audio/mpeg")
        reduction_bytes = source_info["size_bytes"] - output_info["size_bytes"]
        reduction_pct = (
            reduction_bytes / source_info["size_bytes"] * 100 if source_info["size_bytes"] else 0.0
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
            audio_path = await AudioDownloader.download_audio(
                job.input_url or track.audio_url,
                suffix=".source",
                db=db,
                job_id=job.id,
                run_id=job.run_id,
                track_id=track.track_id,
                purpose="discovery_source",
                convert_to_wav=True,
                preserve_channels=True,
            )
            transcript_data = await self._transcriber.transcribe_file(
                audio_path, job_id=job.id, run_id=job.run_id, track_id=track.track_id
            )
            transcript_text = self._coerce_transcript_text(
                (transcript_data or {}).get("transcript", "")
            )
            self._push_stage_result(job, track_job, "transcribing", transcript_data or {})
        if not transcript_text:
            report = self._no_content_report(transcript_data)
            await self._complete(
                db,
                job,
                track_job,
                {
                    "job_id": job.id,
                    "run_id": job.run_id,
                    "job_type": job.job_type,
                    "track_id": track.track_id,
                    "transcription": transcript_data or {},
                    "report": report,
                    "flagged": True,
                },
            )
            return
        if not await self._set_stage(db, job, track_job, "discovering"):
            return
        discovery, content_description = await self._run_discovery(
            track, transcript_text, None, source=(job.job_options or {}).get("source")
        )
        discovery_data = discovery or {}
        track_job.discovery_json = discovery_data
        await DatabaseCommitter.commit_with_retry(db)
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
        platform = await PlatformSettingsProvider.fetch_platform_settings()
        transcript_text = ""
        segments: list[dict] = []
        tmp_path = None
        if job.job_type == "categorization" and job.edited_transcript:
            transcript_text = job.edited_transcript
            transcript_data = {
                "transcript": transcript_text,
                "segments": segments,
                "language": "en",
                "confidence": 1.0,
                "edited": True,
            }
        elif job.job_type in ("audio_tag",):
            if not await self._set_stage(db, job, track_job, "transcribing"):
                return
            tmp_path = await AudioDownloader.download_audio(
                track.audio_url,
                suffix=".wav",
                db=db,
                job_id=job.id,
                run_id=job.run_id,
                track_id=track.track_id,
                purpose="pipeline_source",
                convert_to_wav=True,
                preserve_channels=True,
            )
            transcript_data = await self._transcriber.transcribe_file(
                tmp_path,
                job_id=job.id,
                run_id=job.run_id,
                track_id=track.track_id,
                short_utterance=True,
            )
            transcript_text = self._coerce_transcript_text(
                (transcript_data or {}).get("transcript", "")
            )
            segments = self._coerce_segments((transcript_data or {}).get("segments", []))
        else:
            is_regeneration = job.attempts == 0 and job.result_json is None
            reused_transcript = ""
            if (
                reused_transcript
                and job.job_type != "transcription"
                and (not is_regeneration)
                and (not job.input_url)
            ):
                transcript_data = {
                    "transcript": reused_transcript,
                    "segments": [],
                    "language": "en",
                    "confidence": 1.0,
                }
                transcript_text = reused_transcript
                segments = []
            else:
                if not await self._set_stage(db, job, track_job, "transcribing"):
                    return
                tmp_path = await AudioDownloader.download_audio(
                    job.input_url or track.audio_url,
                    suffix=".wav",
                    db=db,
                    job_id=job.id,
                    run_id=job.run_id,
                    track_id=track.track_id,
                    purpose="pipeline_source",
                    convert_to_wav=True,
                    preserve_channels=True,
                )
                transcript_data = await self._transcriber.transcribe_file(
                    tmp_path, job_id=job.id, run_id=job.run_id, track_id=track.track_id
                )
                transcript_text = self._coerce_transcript_text(
                    (transcript_data or {}).get("transcript", "")
                )
                segments = self._coerce_segments((transcript_data or {}).get("segments", []))
                edited_ref = (job.edited_transcript or "").strip() or reused_transcript
                if edited_ref and transcript_text:

                    def _strip(s):
                        return set(re.sub("[^\\w\\s]", "", s).lower().split())

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
                            restored if restored != transcript_text else transcript_text, edited_ref
                        )
                        if corrected and corrected != transcript_text:
                            transcript_text = corrected
                            if transcript_data and isinstance(transcript_data, dict):
                                transcript_data["transcript"] = corrected
                                transcript_data["restored"] = True
        track_job.updated_at = datetime.utcnow()
        await DatabaseCommitter.commit_with_retry(db)
        if job.job_type in {"pipeline", "categorization", "rebuild"}:
            self._push_stage_result(job, track_job, "transcribing", transcript_data or {})
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
                        str(tag).strip() for tag in tag_data.get("tags", []) if str(tag).strip()
                    ][:2]
            result = OrchestrationResults.audio_tag_result(job, track, transcript_text, suggestions)
            completed = await self._complete(db, job, track_job, result)
            if not completed:
                return
            return
        if job.job_type == "transcription":
            result = OrchestrationResults.transcription_only_result(job, track, transcript_data)
            completed = await self._complete(db, job, track_job, result)
            if not completed:
                return
            return
        if not transcript_text:
            if not await self._set_stage(db, job, track_job, "moderating"):
                return
            moderation = self._no_content_moderation()
            track_job.moderation_json = moderation
            track_job.updated_at = datetime.utcnow()
            await DatabaseCommitter.commit_with_retry(db)
            result = {
                "job_id": job.id,
                "run_id": job.run_id,
                "track_id": track.track_id,
                "backend_id": job.backend_id,
                "job_type": job.job_type,
                "transcription": transcript_data,
                "moderation": moderation,
                "categorization": None,
                "edited_transcript": job.edited_transcript,
                "report": self._no_content_report(transcript_data),
                "flagged": False,
            }
            self._push_stage_result(job, track_job, "moderating", result["report"])
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
        await DatabaseCommitter.commit_with_retry(db)
        self._stage_times[job.id] = self._stage_times.get(job.id, {})
        self._stage_times[job.id]["moderating"] = round(time.time() - stage_start, 3)
        self._push_stage_result(job, track_job, "moderating", moderation)
        categorization = None
        if not moderation.get("flagged"):
            if not await self._set_stage(db, job, track_job, "categorizing"):
                return
            stage_start = time.time()
            categorization = await self._categorizer.categorize(
                transcript=transcript_text,
                segments=segments,
                custom_tags=platform.auto_tag_keywords,
                max_tags=job.max_tags or 8,
                per_track_transcripts={track.track_id: transcript_text},
            )
            track_job.categorization_json = categorization
            track_job.updated_at = datetime.utcnow()
            await DatabaseCommitter.commit_with_retry(db)
            self._stage_times[job.id]["categorizing"] = round(time.time() - stage_start, 3)
            self._push_stage_result(job, track_job, "categorizing", categorization or {})
        discovery_dict = None
        content_description = None
        if (
            transcript_text
            and (not moderation.get("flagged"))
            and (job.job_type in ("pipeline", "categorization", "rebuild"))
        ):
            if await self._set_stage(db, job, track_job, "discovering"):
                discovery_dict, content_description = await self._run_discovery(
                    track, transcript_text, categorization
                )
                track_job.discovery_json = discovery_dict
                track_job.updated_at = datetime.utcnow()
                await DatabaseCommitter.commit_with_retry(db)
                self._push_stage_result(job, track_job, "discovering", discovery_dict or {})
        compressed_audio = None
        if job.job_type == "pipeline":
            if not await self._set_stage(db, job, track_job, "compressing"):
                return
            compressed_audio = await self._produce_pipeline_mp3(job, track, tmp_path, db)
            self._push_stage_result(job, track_job, "compressing", compressed_audio)
        result = {
            "job_id": job.id,
            "run_id": job.run_id,
            "job_type": job.job_type,
            "backend_id": job.backend_id,
            "track_id": track.track_id,
            "source_audio_url": track.audio_url,
            "transcription": transcript_data,
            "moderation": moderation,
            "categorization": categorization,
            "edited_transcript": job.edited_transcript,
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

    @staticmethod
    def _magic_clean_lineage_jobs(db, job: AiJob) -> list[AiJob]:
        return (
            db.query(AiJob)
            .filter(
                AiJob.backend_id == job.backend_id,
                AiJob.track_id == job.track_id,
                AiJob.status == "completed",
                AiJob.job_type.in_(("magic_clean", "magic-clean")),
                AiJob.result_json.isnot(None),
            )
            .order_by(AiJob.completed_at.desc())
            .all()
        )

    @staticmethod
    def _magic_clean_jobs_outside_scope(db, job: AiJob) -> list[AiJob]:
        candidates = (
            db.query(AiJob)
            .filter(
                AiJob.status == "completed",
                AiJob.job_type.in_(("magic_clean", "magic-clean")),
                AiJob.result_json.isnot(None),
            )
            .all()
        )
        return [
            candidate
            for candidate in candidates
            if candidate.backend_id != job.backend_id or candidate.track_id != job.track_id
        ]

    @staticmethod
    def _reject_cross_scope_magic_clean_match(
        candidates: list[AiJob],
        *,
        submitted_url: str | None = None,
        submitted_file_sha256: str | None = None,
        submitted_pcm_sha256: str | None = None,
    ) -> None:
        for candidate in candidates:
            if (
                submitted_url
                and MagicCleanLineageResolver.extract_enhanced_audio_url(candidate.result_json)
                == submitted_url
            ):
                raise MagicCleanLineageError(
                    "Known Magic Clean output belongs to a different track scope"
                )
            options = candidate.job_options if isinstance(candidate.job_options, dict) else {}
            if (
                submitted_file_sha256
                and str(options.get(MAGIC_CLEAN_DELIVERED_FILE_SHA256_KEY) or "")
                == submitted_file_sha256
            ):
                raise MagicCleanLineageError(
                    "Known Magic Clean bytes belong to a different track scope"
                )
            if (
                submitted_pcm_sha256
                and str(options.get(MAGIC_CLEAN_DELIVERED_PCM_SHA256_KEY) or "")
                == submitted_pcm_sha256
            ):
                raise MagicCleanLineageError(
                    "Known Magic Clean audio belongs to a different track scope"
                )

    @staticmethod
    def _known_magic_clean_source_hashes(
        lineage_jobs: list[AiJob], job_ids: set[str]
    ) -> tuple[str | None, str | None]:
        file_hashes: set[str] = set()
        pcm_hashes: set[str] = set()
        for candidate in lineage_jobs:
            if candidate.id not in job_ids:
                continue
            options = candidate.job_options if isinstance(candidate.job_options, dict) else {}
            file_hash = str(options.get(MAGIC_CLEAN_SOURCE_FILE_SHA256_KEY) or "")
            pcm_hash = str(options.get(MAGIC_CLEAN_SOURCE_PCM_SHA256_KEY) or "")
            if file_hash:
                file_hashes.add(file_hash)
            if pcm_hash:
                pcm_hashes.add(pcm_hash)
        if len(file_hashes) > 1 or len(pcm_hashes) > 1:
            raise MagicCleanLineageError(
                "Magic Clean lineage has conflicting canonical source hashes"
            )
        return (next(iter(file_hashes), None), next(iter(pcm_hashes), None))

    async def _prepare_magic_clean_source(
        self, db, job: AiJob, *, submitted_url: str, track_id: str, levels: dict
    ) -> tuple[str, dict[str, object], float]:
        options = dict(job.job_options or {})
        persisted_root = str(options.get(MAGIC_CLEAN_ROOT_URL_KEY) or "").strip()
        persisted_engine = str(options.get(MAGIC_CLEAN_ENGINE_REVISION_KEY) or "").strip()
        lineage_jobs = self._magic_clean_lineage_jobs(db, job)
        outside_scope_jobs = self._magic_clean_jobs_outside_scope(db, job)
        self._reject_cross_scope_magic_clean_match(outside_scope_jobs, submitted_url=submitted_url)
        expected_file_hash = (
            str(options.get(MAGIC_CLEAN_SOURCE_FILE_SHA256_KEY) or "").strip() or None
        )
        expected_pcm_hash = (
            str(options.get(MAGIC_CLEAN_SOURCE_PCM_SHA256_KEY) or "").strip() or None
        )
        if persisted_root:
            if persisted_engine != settings.MAGIC_CLEAN_ENGINE_REVISION:
                raise MagicCleanLineageError(
                    "Magic Clean retry is pinned to an unavailable engine revision"
                )
            if options.get("magic_clean_controls") != dict(levels):
                raise MagicCleanLineageError(
                    "Magic Clean retry controls differ from the pinned request"
                )
            root_url = persisted_root
            parent_job_id = str(options.get(MAGIC_CLEAN_PARENT_JOB_ID_KEY) or "").strip()
            matched_by = "persisted_retry"
            if parent_job_id and (not (expected_file_hash and expected_pcm_hash)):
                recovered_file_hash, recovered_pcm_hash = self._known_magic_clean_source_hashes(
                    lineage_jobs, {parent_job_id}
                )
                expected_file_hash = expected_file_hash or recovered_file_hash
                expected_pcm_hash = expected_pcm_hash or recovered_pcm_hash
        else:
            resolution = MagicCleanLineageResolver.resolve_magic_clean_lineage(
                submitted_url,
                backend_id=str(job.backend_id or ""),
                track_id=track_id,
                jobs=lineage_jobs,
            )
            root_url = resolution.root_url
            parent_job_id = resolution.lineage_job_ids[0] if resolution.lineage_job_ids else ""
            matched_by = resolution.matched_by
            expected_ids = set(resolution.lineage_job_ids) | set(resolution.matched_job_ids)
            expected_file_hash, expected_pcm_hash = self._known_magic_clean_source_hashes(
                lineage_jobs, expected_ids
            )
        options.update(
            {
                MAGIC_CLEAN_ROOT_URL_KEY: root_url,
                MAGIC_CLEAN_PARENT_JOB_ID_KEY: parent_job_id,
                MAGIC_CLEAN_ENGINE_REVISION_KEY: settings.MAGIC_CLEAN_ENGINE_REVISION,
                "magic_clean_match_method": matched_by,
                "magic_clean_controls": dict(levels),
            }
        )
        if expected_file_hash:
            options[MAGIC_CLEAN_SOURCE_FILE_SHA256_KEY] = expected_file_hash
        if expected_pcm_hash:
            options[MAGIC_CLEAN_SOURCE_PCM_SHA256_KEY] = expected_pcm_hash
        job.job_options = options
        await DatabaseCommitter.commit_with_retry(db)
        download_started = time.perf_counter()
        audio_path = await AudioDownloader.download_audio(
            root_url,
            suffix=".audio",
            db=db,
            job_id=job.id,
            run_id=job.run_id,
            track_id=track_id,
            purpose="magic_clean_source",
            convert_to_wav=False,
        )
        file_hash, pcm_hash = await AsyncCompletion.run_blocking_to_completion(
            partial(MagicCleanLineageResolver.magic_clean_artifact_hashes, audio_path)
        )
        self._reject_cross_scope_magic_clean_match(
            outside_scope_jobs, submitted_file_sha256=file_hash, submitted_pcm_sha256=pcm_hash
        )
        if not persisted_root and root_url == submitted_url:
            alias = MagicCleanLineageResolver.resolve_magic_clean_hash_alias(
                submitted_url,
                backend_id=str(job.backend_id or ""),
                track_id=track_id,
                jobs=lineage_jobs,
                submitted_file_sha256=file_hash,
                submitted_pcm_sha256=pcm_hash,
            )
            if alias.is_known_derivative:
                options["magic_clean_submitted_file_sha256"] = file_hash
                options["magic_clean_submitted_pcm_sha256"] = pcm_hash
                TempWorkspace.drop_temp_standalone(audio_path)
                root_url = alias.root_url
                parent_job_id = (
                    alias.lineage_job_ids[0] if alias.lineage_job_ids else alias.matched_job_ids[0]
                )
                expected_ids = set(alias.lineage_job_ids) | set(alias.matched_job_ids)
                expected_file_hash, expected_pcm_hash = self._known_magic_clean_source_hashes(
                    lineage_jobs, expected_ids
                )
                options.update(
                    {
                        MAGIC_CLEAN_ROOT_URL_KEY: root_url,
                        MAGIC_CLEAN_PARENT_JOB_ID_KEY: parent_job_id,
                        "magic_clean_match_method": alias.matched_by,
                    }
                )
                if expected_file_hash:
                    options[MAGIC_CLEAN_SOURCE_FILE_SHA256_KEY] = expected_file_hash
                if expected_pcm_hash:
                    options[MAGIC_CLEAN_SOURCE_PCM_SHA256_KEY] = expected_pcm_hash
                job.job_options = options
                await DatabaseCommitter.commit_with_retry(db)
                audio_path = await AudioDownloader.download_audio(
                    root_url,
                    suffix=".audio",
                    db=db,
                    job_id=job.id,
                    run_id=job.run_id,
                    track_id=track_id,
                    purpose="magic_clean_source",
                    convert_to_wav=False,
                )
                file_hash, pcm_hash = await AsyncCompletion.run_blocking_to_completion(
                    partial(MagicCleanLineageResolver.magic_clean_artifact_hashes, audio_path)
                )
        if expected_file_hash and file_hash != expected_file_hash:
            TempWorkspace.drop_temp_standalone(audio_path)
            raise MagicCleanLineageError(
                "Magic Clean canonical source bytes changed after lineage was pinned"
            )
        if expected_pcm_hash and pcm_hash != expected_pcm_hash:
            TempWorkspace.drop_temp_standalone(audio_path)
            raise MagicCleanLineageError(
                "Magic Clean canonical decoded audio changed after lineage was pinned"
            )
        options.update(
            {
                MAGIC_CLEAN_ROOT_URL_KEY: root_url,
                MAGIC_CLEAN_PARENT_JOB_ID_KEY: parent_job_id,
                MAGIC_CLEAN_SOURCE_FILE_SHA256_KEY: file_hash,
                MAGIC_CLEAN_SOURCE_PCM_SHA256_KEY: pcm_hash,
            }
        )
        job.job_options = options
        await DatabaseCommitter.commit_with_retry(db)
        return (audio_path, options, round(time.perf_counter() - download_started, 3))

    @staticmethod
    def _magic_clean_reuse_candidate(
        lineage_jobs: list[AiJob], source_metadata: dict[str, object], levels: dict
    ) -> AiJob | None:
        for candidate in lineage_jobs:
            options = candidate.job_options if isinstance(candidate.job_options, dict) else {}
            if options.get("magic_clean_validated") is not True:
                continue
            if (
                options.get(MAGIC_CLEAN_ROOT_URL_KEY)
                != source_metadata.get(MAGIC_CLEAN_ROOT_URL_KEY)
                or options.get(MAGIC_CLEAN_SOURCE_FILE_SHA256_KEY)
                != source_metadata.get(MAGIC_CLEAN_SOURCE_FILE_SHA256_KEY)
                or options.get(MAGIC_CLEAN_SOURCE_PCM_SHA256_KEY)
                != source_metadata.get(MAGIC_CLEAN_SOURCE_PCM_SHA256_KEY)
                or (
                    options.get(MAGIC_CLEAN_ENGINE_REVISION_KEY)
                    != settings.MAGIC_CLEAN_ENGINE_REVISION
                )
                or (options.get("magic_clean_controls") != dict(levels))
            ):
                continue
            result = candidate.result_json
            quality = result.get("quality") if isinstance(result, dict) else None
            if not isinstance(quality, dict):
                continue
            if not MagicCleanLineageResolver.extract_enhanced_audio_url(result):
                continue
            delivered_file_hash = str(options.get(MAGIC_CLEAN_DELIVERED_FILE_SHA256_KEY) or "")
            delivered_pcm_hash = str(options.get(MAGIC_CLEAN_DELIVERED_PCM_SHA256_KEY) or "")
            if not delivered_file_hash or not delivered_pcm_hash:
                continue
            return candidate
        return None

    async def _try_reuse_magic_clean_artifact(
        self, db, job: AiJob, storage, candidate: AiJob, source_metadata: dict[str, object]
    ) -> dict | None:
        options = candidate.job_options if isinstance(candidate.job_options, dict) else {}
        prior_url = MagicCleanLineageResolver.extract_enhanced_audio_url(candidate.result_json)
        expected_file_hash = str(options.get(MAGIC_CLEAN_DELIVERED_FILE_SHA256_KEY) or "")
        expected_pcm_hash = str(options.get(MAGIC_CLEAN_DELIVERED_PCM_SHA256_KEY) or "")
        started_total = time.perf_counter()
        reused_path = ""
        remote_key = ""
        upload_attempted = False
        try:
            started = time.perf_counter()
            reused_path = await AudioDownloader.download_audio(
                prior_url,
                suffix=".mp3",
                db=db,
                job_id=job.id,
                run_id=job.run_id,
                track_id=job.track_id,
                purpose="magic_clean_validated_reuse",
                convert_to_wav=False,
            )
            download_seconds = round(time.perf_counter() - started, 3)
            started = time.perf_counter()
            file_hash, pcm_hash = await AsyncCompletion.run_blocking_to_completion(
                partial(MagicCleanLineageResolver.magic_clean_artifact_hashes, reused_path)
            )
            hash_seconds = round(time.perf_counter() - started, 3)
            if file_hash != expected_file_hash or pcm_hash != expected_pcm_hash:
                logger.warning(
                    "Validated Magic Clean reuse artifact changed for job=%s", candidate.id
                )
                return None
            remote_key = storage.key("enhanced", f"{job.id}.mp3")
            started = time.perf_counter()
            upload_attempted = True
            enhanced_url = await AsyncCompletion.run_blocking_to_completion(
                partial(
                    storage.upload_file,
                    reused_path,
                    remote_key,
                    "audio/mpeg",
                    checksum_sha256=file_hash,
                )
            )
            upload_seconds = round(time.perf_counter() - started, 3)
            prior_result = candidate.result_json
            quality = prior_result["quality"]
            return {
                "b2_key": remote_key,
                "enhanced_url": enhanced_url,
                "bucket_name": storage.bucket_name,
                "quality_score": quality["quality_score"],
                "snr_db": quality["snr_db"],
                "peak_db": quality["peak_db"],
                "lufs": quality["lufs"],
                "clipping_detected": quality["clipping_detected"],
                "source_file_sha256": source_metadata[MAGIC_CLEAN_SOURCE_FILE_SHA256_KEY],
                "source_pcm_sha256": source_metadata[MAGIC_CLEAN_SOURCE_PCM_SHA256_KEY],
                "delivered_file_sha256": file_hash,
                "delivered_pcm_sha256": pcm_hash,
                "engine_revision": settings.MAGIC_CLEAN_ENGINE_REVISION,
                "stage_times": {
                    "reuse_download": download_seconds,
                    "reuse_hash": hash_seconds,
                    "reuse_upload": upload_seconds,
                    "total": round(time.perf_counter() - started_total, 3),
                },
            }
        except asyncio.CancelledError:
            if upload_attempted:
                await self._cleanup_magic_clean_artifact(
                    db,
                    job,
                    storage,
                    {
                        "b2_key": remote_key,
                        "bucket_name": storage.bucket_name,
                        "delivered_file_sha256": expected_file_hash,
                    },
                    reason="cancelled_during_reuse_upload",
                )
            raise
        except Exception as exc:
            if upload_attempted:
                await self._cleanup_magic_clean_artifact(
                    db,
                    job,
                    storage,
                    {
                        "b2_key": remote_key,
                        "bucket_name": storage.bucket_name,
                        "delivered_file_sha256": expected_file_hash,
                    },
                    reason="reuse_upload_failed",
                )
            logger.warning(
                "Could not reuse validated Magic Clean artifact for job=%s: %s",
                candidate.id,
                type(exc).__name__,
            )
            return None
        finally:
            if reused_path:
                TempWorkspace.drop_temp_standalone(reused_path)

    @staticmethod
    def _validate_magic_clean_enhancement(
        enhancement: object,
        source_metadata: dict[str, object],
        *,
        expected_key: str,
        expected_bucket: str,
        expected_url: str,
    ) -> dict:
        if not isinstance(enhancement, dict):
            raise RuntimeError("Magic Clean deployment returned an invalid result")
        for name in (
            "enhanced_url",
            "b2_key",
            "bucket_name",
            "source_file_sha256",
            "source_pcm_sha256",
            "delivered_file_sha256",
            "delivered_pcm_sha256",
            "engine_revision",
        ):
            value = enhancement.get(name)
            if not isinstance(value, str) or not value.strip():
                raise RuntimeError(f"Magic Clean result is missing {name}")
        if enhancement["b2_key"] != expected_key:
            raise RuntimeError("Magic Clean result used an unexpected storage key")
        if enhancement["bucket_name"] != expected_bucket:
            raise RuntimeError("Magic Clean result used an unexpected storage bucket")
        if enhancement["enhanced_url"] != expected_url:
            raise RuntimeError("Magic Clean result used an unexpected artifact URL")
        for name in (
            "source_file_sha256",
            "source_pcm_sha256",
            "delivered_file_sha256",
            "delivered_pcm_sha256",
        ):
            digest = enhancement[name].strip().lower()
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise RuntimeError(f"Magic Clean result has invalid {name}")
        for name in ("quality_score", "snr_db", "peak_db", "lufs"):
            value = enhancement.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RuntimeError(f"Magic Clean result has invalid {name}")
            if not math.isfinite(float(value)):
                raise RuntimeError(f"Magic Clean result has non-finite {name}")
        if enhancement.get("clipping_detected") is not False:
            raise RuntimeError("Magic Clean delivered validation reported clipping")
        stage_times = enhancement.get("stage_times")
        if not isinstance(stage_times, dict) or not stage_times:
            raise RuntimeError("Magic Clean result is missing stage timings")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or (not math.isfinite(float(value)))
            or (float(value) < 0)
            for value in stage_times.values()
        ):
            raise RuntimeError("Magic Clean stage timings are invalid")
        if enhancement["engine_revision"] != settings.MAGIC_CLEAN_ENGINE_REVISION:
            raise RuntimeError("Magic Clean deployment used the wrong engine revision")
        for result_name, metadata_name in (
            ("source_file_sha256", MAGIC_CLEAN_SOURCE_FILE_SHA256_KEY),
            ("source_pcm_sha256", MAGIC_CLEAN_SOURCE_PCM_SHA256_KEY),
        ):
            if enhancement[result_name] != source_metadata.get(metadata_name):
                raise RuntimeError("Magic Clean deployment processed the wrong source")
        return enhancement

    async def _record_magic_clean_cleanup_tombstone(
        self, job: AiJob, enhancement: object, *, reason: str, error_type: str
    ) -> None:
        if not isinstance(enhancement, dict):
            return
        key = enhancement.get("b2_key")
        if not isinstance(key, str) or not key.strip():
            return
        created_at = datetime.now(UTC)
        tombstone = {
            "b2_key": key,
            "bucket_name": str(enhancement.get("bucket_name") or ""),
            "run_id": str(getattr(job, "run_id", "") or ""),
            "job_attempt": int(getattr(job, "attempts", 0) or 0),
            "delivered_file_sha256": str(enhancement.get("delivered_file_sha256") or ""),
            "reason": reason,
            "created_at": created_at.isoformat(),
            "not_before": (
                created_at + timedelta(seconds=settings.MAGIC_CLEAN_CLEANUP_GRACE_SECONDS)
            ).isoformat(),
            "last_error_type": error_type,
        }
        cleanup_db = DatabaseRuntime.SessionLocal()
        try:
            persisted_job = (
                cleanup_db.query(AiJob)
                .filter(AiJob.id == job.id, AiJob.run_id == getattr(job, "run_id", None))
                .with_for_update()
                .first()
            )
            if persisted_job is None:
                raise RuntimeError("Magic Clean cleanup owner job is unavailable")
            persisted_options = dict(persisted_job.job_options or {})
            persisted_options["magic_clean_cleanup_tombstone"] = tombstone
            persisted_job.job_options = persisted_options
            job.job_options = dict(persisted_options)
            await AsyncCompletion.run_awaitable_to_completion(
                DatabaseCommitter.commit_with_retry(cleanup_db)
            )
        except BaseException:
            cleanup_db.rollback()
            raise
        finally:
            cleanup_db.close()

    async def _cleanup_magic_clean_artifact(
        self,
        db,
        job: AiJob,
        storage,
        enhancement: object,
        *,
        reason: str,
        defer_deletion: bool = False,
    ) -> None:
        if not isinstance(enhancement, dict):
            return
        key = enhancement.get("b2_key")
        if not isinstance(key, str) or not key.strip():
            return
        if defer_deletion:
            await self._record_magic_clean_cleanup_tombstone(
                job, enhancement, reason=reason, error_type="DeferredOwnershipCheck"
            )
            return
        try:
            await AsyncCompletion.run_blocking_to_completion(partial(storage.delete_object, key))
        except asyncio.CancelledError:
            await self._record_magic_clean_cleanup_tombstone(
                job, enhancement, reason=reason, error_type="CleanupCancelled"
            )
            raise
        except Exception as exc:
            await self._record_magic_clean_cleanup_tombstone(
                job, enhancement, reason=reason, error_type=type(exc).__name__
            )
            logger.error("Magic Clean artifact cleanup requires reconciliation for job=%s", job.id)

    async def _process_magic_clean(self, job: AiJob, track_job: AiTrackJob, db):
        if not job.track_id:
            raise ValueError("track_id is required for magic_clean")
        submitted_url = str(job.input_url or "").strip()
        if not submitted_url:
            raise ValueError("audio_url is required for magic_clean")
        levels = self._magic_clean_levels(job)
        if not await self._set_stage(db, job, track_job, "downloading", levels):
            return
        track = self._track_from_job(job)
        audio_path, source_metadata, download_seconds = await self._prepare_magic_clean_source(
            db, job, submitted_url=submitted_url, track_id=track.track_id, levels=levels
        )
        if not await self._set_stage(db, job, track_job, "separating", levels):
            return
        if not await self._set_stage(db, job, track_job, "enhancing", levels):
            return
        storage = StorageContexts.storage_for_job(job)
        self._require_magic_clean_storage_lifetime(storage)
        expected_key = storage.key("enhanced", f"{job.id}.mp3")
        expected_artifact = {"b2_key": expected_key, "bucket_name": storage.bucket_name}
        await self._record_magic_clean_cleanup_tombstone(
            job, expected_artifact, reason="upload_pending", error_type="ProvisionalArtifact"
        )
        reuse_candidate = self._magic_clean_reuse_candidate(
            self._magic_clean_lineage_jobs(db, job), source_metadata, levels
        )
        enhancement = None
        if reuse_candidate is not None:
            enhancement = await self._try_reuse_magic_clean_artifact(
                db, job, storage, reuse_candidate, source_metadata
            )
        if enhancement is None:
            if self._magic_clean_handle is None:
                raise RuntimeError("magic_clean Ray deployment is unavailable")
            try:
                remote_response = self._magic_clean_handle.enhance.remote(
                    audio_url=str(source_metadata[MAGIC_CLEAN_ROOT_URL_KEY]),
                    track_id=track.track_id,
                    job_id=job.id,
                    ai_job_id=job.id,
                    ai_run_id=job.run_id,
                    speech=levels["speech"],
                    music=levels["music"],
                    background=levels["background"],
                    cut_silence=levels["cut_silence"],
                    storage_context=storage.context.model_dump(mode="json"),
                    expected_source_file_sha256=str(
                        source_metadata[MAGIC_CLEAN_SOURCE_FILE_SHA256_KEY]
                    ),
                    expected_source_pcm_sha256=str(
                        source_metadata[MAGIC_CLEAN_SOURCE_PCM_SHA256_KEY]
                    ),
                )
                cancel_remote = getattr(remote_response, "cancel", None)
                enhancement = await AsyncCompletion.run_awaitable_to_completion(
                    remote_response, on_cancel=cancel_remote if callable(cancel_remote) else None
                )
            except asyncio.CancelledError:
                await self._cleanup_magic_clean_artifact(
                    db, job, storage, expected_artifact, reason="cancelled_during_remote_processing"
                )
                raise
            except Exception:
                await self._cleanup_magic_clean_artifact(
                    db, job, storage, expected_artifact, reason="remote_processing_failed"
                )
                raise
        try:
            enhancement = self._validate_magic_clean_enhancement(
                enhancement,
                source_metadata,
                expected_key=expected_key,
                expected_bucket=storage.bucket_name,
                expected_url=storage._public_url(expected_key),
            )
        except Exception:
            await self._cleanup_magic_clean_artifact(
                db, job, storage, expected_artifact, reason="invalid_remote_result"
            )
            raise
        try:
            mixing_ready = await self._set_stage(db, job, track_job, "mixing", levels)
        except asyncio.CancelledError:
            await self._cleanup_magic_clean_artifact(
                db, job, storage, enhancement, reason="cancelled_during_mixing_transition"
            )
            raise
        except Exception:
            await self._cleanup_magic_clean_artifact(
                db, job, storage, enhancement, reason="mixing_transition_failed"
            )
            raise
        if not mixing_ready:
            await self._cleanup_magic_clean_artifact(
                db, job, storage, enhancement, reason="cancelled_after_upload"
            )
            return
        TempWorkspace.drop_temp_standalone(audio_path)
        try:
            finalizing_ready = await self._set_stage(db, job, track_job, "finalizing", levels)
        except asyncio.CancelledError:
            await self._cleanup_magic_clean_artifact(
                db, job, storage, enhancement, reason="cancelled_during_finalizing_transition"
            )
            raise
        except Exception:
            await self._cleanup_magic_clean_artifact(
                db, job, storage, enhancement, reason="finalizing_transition_failed"
            )
            raise
        if not finalizing_ready:
            await self._cleanup_magic_clean_artifact(
                db, job, storage, enhancement, reason="cancelled_before_completion"
            )
            return
        options = dict(job.job_options or {})
        options.pop("magic_clean_cleanup_tombstone", None)
        options.update(
            {
                MAGIC_CLEAN_DELIVERED_FILE_SHA256_KEY: enhancement["delivered_file_sha256"],
                MAGIC_CLEAN_DELIVERED_PCM_SHA256_KEY: enhancement["delivered_pcm_sha256"],
                "magic_clean_validated": True,
            }
        )
        if reuse_candidate is not None and "reuse_download" in enhancement["stage_times"]:
            options["magic_clean_reused_from_job_id"] = reuse_candidate.id
        job.job_options = options
        stage_times = {"downloading": download_seconds, **enhancement["stage_times"]}
        result = {
            "job_id": job.id,
            "run_id": job.run_id,
            "track_id": track.track_id,
            "backend_id": job.backend_id,
            "job_type": job.job_type,
            "transcription": {},
            "moderation": {},
            "categorization": None,
            "enhanced": True,
            "enhanced_audio": {
                "audio_url": enhancement["enhanced_url"],
                "b2_key": enhancement["b2_key"],
                "bucket_name": enhancement["bucket_name"],
                "backend_id": job.backend_id,
            },
            "quality": {
                "quality_score": enhancement["quality_score"],
                "snr_db": enhancement["snr_db"],
                "peak_db": enhancement["peak_db"],
                "lufs": enhancement["lufs"],
                "clipping_detected": enhancement["clipping_detected"],
            },
            "stage_times": stage_times,
        }
        try:
            completed = await self._complete(db, job, track_job, result)
        except asyncio.CancelledError:
            await self._cleanup_magic_clean_artifact(
                db,
                job,
                storage,
                enhancement,
                reason="cancelled_during_completion_transition",
                defer_deletion=True,
            )
            raise
        except Exception:
            await self._cleanup_magic_clean_artifact(
                db,
                job,
                storage,
                enhancement,
                reason="completion_transition_failed",
                defer_deletion=True,
            )
            raise
        if not completed:
            await self._cleanup_magic_clean_artifact(
                db,
                job,
                storage,
                enhancement,
                reason="completion_lost_to_cancellation",
                defer_deletion=True,
            )

    def _resolve_reconstruction_reference_url(
        self, db, job: AiJob, track_id: str, submitted_url: str, changes: list[dict]
    ) -> tuple[str, int]:
        options = dict(job.job_options or {})
        requested_intervals = OrchestrationResults._change_intervals(changes)
        persisted_root = str(options.get("voice_reference_audio_url") or "").strip()
        persisted_intervals = OrchestrationResults._change_intervals(
            options.get("voice_reference_intervals")
        )
        if persisted_root and requested_intervals and (requested_intervals == persisted_intervals):
            return (persisted_root, int(persisted_root != submitted_url))
        lineage_jobs = (
            db.query(AiJob)
            .filter(
                AiJob.backend_id == job.backend_id,
                AiJob.track_id == track_id,
                AiJob.status == "completed",
                AiJob.job_type.in_(tuple(RECONSTRUCTION_LINEAGE_JOB_TYPES)),
                AiJob.result_json.isnot(None),
            )
            .order_by(AiJob.completed_at.desc())
            .all()
        )
        resolved_url, hops = OrchestrationResults.resolve_reconstruction_reference_url(
            submitted_url,
            backend_id=str(job.backend_id or ""),
            track_id=track_id,
            changes=changes,
            jobs=lineage_jobs,
            exclude_job_id=job.id,
        )
        job.job_options = {
            **options,
            "voice_reference_audio_url": resolved_url,
            "voice_reference_intervals": [
                {"segment_start": start, "segment_end": end} for start, end in requested_intervals
            ],
        }
        return (resolved_url, hops)

    async def _download_reconstruction_root(
        self,
        db,
        job: AiJob,
        *,
        track_id: str,
        submitted_url: str,
        changes: list[dict],
        purpose: str,
    ) -> str | None:
        reference_url, lineage_hops = self._resolve_reconstruction_reference_url(
            db, job, track_id, submitted_url, changes
        )
        if not reference_url or reference_url == submitted_url:
            return None
        try:
            reference_path = await AudioDownloader.download_audio(
                reference_url,
                suffix=".wav",
                db=db,
                job_id=job.id,
                run_id=job.run_id,
                track_id=track_id,
                purpose=purpose,
                convert_to_wav=True,
            )
        except Exception:
            logger.error(
                "Immutable reconstruction root is unavailable for track=%s; refusing recursive reconstruction",
                track_id,
            )
            raise
        _recon_logger.info(
            "REFERENCE_LINEAGE | track=%s | hops=%d | immutable=true", track_id, lineage_hops
        )
        return reference_path

    async def _process_reconstruct(self, job: AiJob, track_job: AiTrackJob, db):
        changes, same_speaker = self._coerce_reconstruct_payload(job.custom_tags or {})
        if not changes:
            raise ValueError(
                "reconstruct requires changes[] with segment_start/segment_end/new_text"
            )
        track = self._track_from_job(job)
        submitted_audio_url = str(job.input_url or track.audio_url or "").strip()
        audio_path = await AudioDownloader.download_audio(
            submitted_audio_url,
            suffix=".wav",
            db=db,
            job_id=job.id,
            run_id=job.run_id,
            track_id=track.track_id,
            purpose="reconstruct_source",
            convert_to_wav=True,
        )
        immutable_root_path = await self._download_reconstruction_root(
            db,
            job,
            track_id=track.track_id,
            submitted_url=submitted_audio_url,
            purpose="reconstruct_reference",
            changes=changes,
        )
        reconstruction_audio_path = immutable_root_path or audio_path
        voice_reference_audio_path = reconstruction_audio_path if same_speaker else None
        if not await self._set_stage(db, job, track_job, "reconstructing"):
            return
        rebuilt = await self._synthesizer.reconstruct_segments(
            voice_reference_audio_path=voice_reference_audio_path,
            original_audio_path=reconstruction_audio_path,
            changes=changes,
            storage=StorageContexts.storage_for_job(job),
            same_speaker=same_speaker,
            job_id=job.id,
            run_id=job.run_id,
            track_id=track.track_id,
        )
        result = {
            "job_id": job.id,
            "run_id": job.run_id,
            "track_id": track.track_id,
            "backend_id": job.backend_id,
            "job_type": job.job_type,
            "transcription": {},
            "moderation": {},
            "categorization": None,
            "rebuilt_audio": {
                "audio_url": rebuilt.audio_url,
                "b2_key": rebuilt.b2_key,
                "duration": rebuilt.duration,
                "bucket_name": rebuilt.bucket_name,
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
        _, same_speaker = self._coerce_reconstruct_payload(job.custom_tags or {})
        track = self._track_from_job(job)
        submitted_audio_url = str(job.input_url or track.audio_url or "").strip()
        audio_path = await AudioDownloader.download_audio(
            submitted_audio_url,
            suffix=".wav",
            db=db,
            job_id=job.id,
            run_id=job.run_id,
            track_id=track.track_id,
            purpose="edit_transcript_source",
            convert_to_wav=True,
        )
        transcript_data = await self._transcriber.transcribe_file(
            audio_path, job_id=job.id, run_id=job.run_id, track_id=track.track_id
        )
        original_text = self._coerce_transcript_text((transcript_data or {}).get("transcript", ""))
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
        immutable_root_path = await self._download_reconstruction_root(
            db,
            job,
            track_id=track.track_id,
            submitted_url=submitted_audio_url,
            changes=changes,
            purpose="edit_transcript_reference",
        )
        reconstruction_audio_path = immutable_root_path or audio_path
        if not await self._set_stage(db, job, track_job, "reconstructing"):
            return
        rebuilt = await self._synthesizer.reconstruct_segments(
            voice_reference_audio_path=reconstruction_audio_path if same_speaker else None,
            original_audio_path=reconstruction_audio_path,
            changes=changes,
            storage=StorageContexts.storage_for_job(job),
            same_speaker=same_speaker,
            job_id=job.id,
            run_id=job.run_id,
            track_id=track.track_id,
        )
        result = {
            "job_id": job.id,
            "run_id": job.run_id,
            "track_id": track.track_id,
            "backend_id": job.backend_id,
            "job_type": job.job_type,
            "transcription": transcript_data,
            "moderation": {},
            "categorization": None,
            "edited_transcript": edited,
            "rebuilt_audio": {
                "audio_url": rebuilt.audio_url,
                "b2_key": rebuilt.b2_key,
                "duration": rebuilt.duration,
                "bucket_name": rebuilt.bucket_name,
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
            return ([], True)
        raw_changes = value.get("changes")
        same_speaker = bool(value.get("same_speaker", True))
        if not isinstance(raw_changes, list):
            return ([], same_speaker)
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
            changes.append(
                {
                    "segment_start": start,
                    "segment_end": end,
                    "new_text": text,
                    "original_text": original_text,
                    "is_deletion": is_deletion,
                }
            )
        return (changes, same_speaker)
