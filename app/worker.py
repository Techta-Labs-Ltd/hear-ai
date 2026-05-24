import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import asyncio

import json
import re
import os
import traceback
from datetime import datetime
from functools import partial

import sentry_sdk
from app.config import settings
from app.core.db_gate import commit_with_retry
from app.core.gpu import gpu, ml_job_lock
from app.core.audio_utils import convert_wav_file_to_mp3
from app.core.downloader import download_audio
from app.core.hear_temp import (
    cleanup_job_temp,
    drop_temp_standalone,
    hear_temp_directory,
    sweep_tracked_temp_files,
)
from app.core.async_speed_upload import upload_pipeline_speed_layers
from app.core.pipeline_speeds import merge_speed_multipliers, parse_speed_multiplier_csv
from app.core.storage import storage
from app.core.track_b2_cleanup import cleanup_track_ai_b2_assets
from app.core.recording_fetcher import effective_transcript_text, fetch_track
from app.core.platform_settings import fetch_platform_settings
from app.models.database import SessionLocal, AiJob, AiTrackJob
from app.realtime.broadcaster import manager
from app.services.callback import callback_service
from app.models.discovery import coerce_discovery_source
from app.services.discovery import discovery_result_bundle, discovery_service
from app.services.llm_service import llm_service

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
        self._global_limit = asyncio.Semaphore(max(1, settings.MAX_CONCURRENT_JOBS))
        self._pipeline_limit = asyncio.Semaphore(max(1, settings.MAX_CONCURRENT_PIPELINE_JOBS))
        self._magic_clean_limit = asyncio.Semaphore(max(1, settings.MAX_CONCURRENT_MAGIC_CLEAN_JOBS))

    async def start(self):
        self._running = True
        print(
            f"[WORKER] single-queue execution; limits jobs={settings.MAX_CONCURRENT_JOBS} "
            f"pipeline={settings.MAX_CONCURRENT_PIPELINE_JOBS} magic_clean={settings.MAX_CONCURRENT_MAGIC_CLEAN_JOBS} "
            f"gpu={settings.MAX_CONCURRENT_GPU_JOBS}"
        )
        await self._recover_jobs()
        try:
            summary = sweep_tracked_temp_files()
            total = summary["by_job"] + summary["by_age"] + summary["orphan_fs"]
            if total:
                print(
                    f"[TEMP] Startup removed {total} file(s) "
                    f"(by_job={summary['by_job']}, by_age={summary['by_age']}, "
                    f"orphan_fs={summary['orphan_fs']}, "
                    f"bytes_freed={summary['bytes_freed']}) under {hear_temp_directory()}"
                )
        except Exception as exc:
            print(f"[TEMP] Startup sweep failed: {exc}")
        asyncio.create_task(self._retry_undelivered_callbacks())
        asyncio.create_task(self._temp_sweep_loop())
        self._loop_task = asyncio.create_task(self._loop())

    async def stop(self):
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
        if self._inflight:
            await asyncio.gather(*self._inflight, return_exceptions=True)

    def enqueue(self, job_id: str, run_id: str | None = None):
        self._queue.put_nowait((job_id, run_id))

    async def _recover_jobs(self):
        db = SessionLocal()
        try:
            rows = (
                db.query(AiJob.id, AiJob.run_id)
                .filter(
                    AiJob.status.in_(["queued", "running"]),
                    AiJob.attempts < settings.JOB_MAX_RETRIES,
                )
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
                for r in rows:
                    self.enqueue(r[0], r[1])
                print(f"[WORKER] Recovered {len(rows)} jobs")
        finally:
            db.close()

    async def _retry_undelivered_callbacks(self):
        await asyncio.sleep(10)
        while self._running:
            db = SessionLocal()
            try:
                pending_ids = [
                    r[0]
                    for r in db.query(AiJob.id).filter(
                        AiJob.status.in_(["completed", "failed"]),
                        AiJob.callback_url.isnot(None),
                        AiJob.callback_delivered == False,
                    ).all()
                ]
            finally:
                db.close()
            for jid in pending_ids:
                async with ml_job_lock:
                    rdb = SessionLocal()
                    try:
                        job = (
                            rdb.query(AiJob)
                            .filter(AiJob.id == jid)
                            .with_for_update(skip_locked=True)
                            .first()
                        )
                        if not job or job.callback_delivered or job.status not in ("completed", "failed"):
                            rdb.rollback()
                            continue
                        payload = self._build_result_payload(job)
                        if not payload:
                            rdb.rollback()
                            continue
                        delivered = await callback_service.send(job.callback_url, payload)
                        if delivered:
                            job.callback_delivered = True
                            await commit_with_retry(rdb)
                        else:
                            rdb.rollback()
                    finally:
                        rdb.close()
            await asyncio.sleep(max(15, settings.CALLBACK_RETRY_POLL_SECONDS))

    async def _temp_sweep_loop(self):
        await asyncio.sleep(120)
        interval = max(300, int(settings.HEAR_TEMP_SWEEP_INTERVAL_SECONDS))
        while self._running:
            try:
                summary = await asyncio.to_thread(sweep_tracked_temp_files)
                total = summary["by_job"] + summary["by_age"] + summary["orphan_fs"]
                if total:
                    print(
                        f"[TEMP] Removed {total} file(s) "
                        f"(by_job={summary['by_job']}, by_age={summary['by_age']}, "
                        f"orphan_fs={summary['orphan_fs']}, "
                        f"bytes_freed={summary['bytes_freed']})"
                    )
            except Exception as exc:
                print(f"[TEMP] Sweep failed: {exc}")
            await asyncio.sleep(interval)

    def _build_result_payload(self, job: AiJob) -> dict | None:
        if job.status not in ("completed", "failed"):
            return None
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
            payload = {
                "job_id": job.id,
                "run_id": job.run_id,
                "track_id": job.track_id,
                "job_type": job.job_type,
                "status": "completed",
                "result": result_obj,
                "error": None,
            }
            if job.job_type == "audio_tag" and isinstance(result_obj, dict):
                for key in ("tags", "categories", "media_file_id", "type"):
                    if key in result_obj:
                        payload[key] = result_obj[key]
            return payload
        if job.status == "failed":
            return {
                "job_id": job.id,
                "run_id": job.run_id,
                "track_id": job.track_id,
                "job_type": job.job_type,
                "status": "failed",
                "result": None,
                "error": job.error or "unknown",
            }
        return None

    async def _deliver_job_callback(self, job_id: str) -> None:
        db = SessionLocal()
        try:
            job = db.query(AiJob).filter(AiJob.id == job_id).first()
            if not job or not job.callback_url:
                return
            payload = self._build_result_payload(job)
            if not payload:
                return
            delivered = await callback_service.send(job.callback_url, payload)
            job.callback_delivered = delivered
            await commit_with_retry(db)
        finally:
            db.close()

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
        return effective_transcript_text(value)

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

    def _job_options_dict(self, job: AiJob) -> dict:
        raw = getattr(job, "job_options", None)
        return raw if isinstance(raw, dict) else {}

    @staticmethod
    def _audio_tag_result_field(tag_type: str | None) -> str:
        t = (tag_type or "track").strip().lower()
        if t in ("category", "categories"):
            return "categories"
        return "tags"

    @staticmethod
    def _extract_spoken_tags(transcript_text: str) -> list[str]:
        spoken = re.sub(r"[.!?]+$", "", (transcript_text or "").strip())
        if not spoken:
            return []
        return [t.strip() for t in re.split(r",\s*|\s+and\s+|\n+", spoken) if t.strip()]

    def _coerce_job_speed_multipliers(self, raw) -> list[float] | None:
        if not isinstance(raw, list):
            return None
        out: list[float] = []
        for x in raw:
            try:
                out.append(float(x))
            except (TypeError, ValueError):
                continue
        return out or None

    @staticmethod
    def _categorization_hint(categorization: dict | None) -> str:
        if not isinstance(categorization, dict):
            return ""
        parts: list[str] = []
        tags = categorization.get("tags") or []
        cats = categorization.get("categories") or []
        if isinstance(tags, list):
            parts.extend(str(t) for t in tags[:14])
        if isinstance(cats, list):
            parts.extend(str(c) for c in cats[:10])
        return ", ".join(parts)[:500]

    @staticmethod
    def _tags_hint(tags) -> str:
        if not tags:
            return ""
        parts: list[str] = []
        for t in tags[:24]:
            if isinstance(t, str):
                parts.append(t)
            elif isinstance(t, dict) and t.get("name"):
                parts.append(str(t["name"]))
        return ", ".join(parts)[:500]

    async def _llm_instruction_speeds(self, job: AiJob) -> list[float]:
        opts = self._job_options_dict(job)
        ins = opts.get("playback_instruction")
        if not isinstance(ins, str) or not ins.strip():
            return []
        if not llm_service.is_available:
            return []
        return await asyncio.to_thread(llm_service.resolve_playback_instruction_speeds, ins.strip())

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
        track_source = coerce_discovery_source(
            source,
            getattr(track, "source", None),
            track_category if isinstance(track_category, str) else None,
        ) or None
        profile = await discovery_service.build_profile(
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
        return discovery_result_bundle(profile, duration_seconds=duration, source=track_source)

    async def _upload_speed_layers(
        self,
        *,
        track_id: str,
        job: AiJob,
        source_path: str,
        speed_list: list[float],
        bitrate_kbps: int | None = None,
    ) -> list[dict]:
        return await upload_pipeline_speed_layers(
            track_id=track_id,
            job_id=job.id,
            run_id=job.run_id,
            source_path=source_path,
            speed_list=speed_list,
            bitrate_kbps=bitrate_kbps,
        )

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
        await commit_with_retry(db)
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

    def _job_completed_broadcast(
        self, job: AiJob, track_job: AiTrackJob, result: dict
    ) -> dict:
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

    async def _complete(self, db, job: AiJob, track_job: AiTrackJob, result: dict) -> bool:
        if not self._run_is_current(db, job.id, job.run_id):
            return False
        now = datetime.utcnow()
        job_id = job.id
        run_id = job.run_id
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
            cleanup_job_temp(db, job_id, run_id)
            await commit_with_retry(db)
        except Exception as exc:
            print(f"[TEMP] cleanup_job_temp on complete failed for {job_id}: {exc}")
        await manager.broadcast(
            job.id, self._job_completed_broadcast(job, track_job, result)
        )
        return True

    async def _process_magic_clean(self, job: AiJob, track_job: AiTrackJob, db):
        if not job.track_id:
            raise ValueError("track_id is required for magic_clean")
        track = await self._fetch_track_with_retry(job.track_id)
        include_enhanced_cleanup = not track.is_enhanced
        b2_deleted = cleanup_track_ai_b2_assets(storage, track, include_enhanced=include_enhanced_cleanup)

        enhanced: dict = {}
        speed_source: str | None = None
        cleanup_paths: list[str] = []

        if track.is_enhanced:
            enhanced[track.track_id] = {
                "enhanced_url": track.audio_url,
                "quality_score": track.quality_score,
                "snr_db": track.snr_db,
            }
            speed_source = await download_audio(
                track.audio_url,
                suffix=None,
                db=db,
                job_id=job.id,
                run_id=job.run_id,
                track_id=track.track_id,
                purpose="magic_clean_enhanced_source",
            )
            cleanup_paths.append(speed_source)
        else:
            local_path = await download_audio(
                track.audio_url,
                suffix=".wav",
                db=db,
                job_id=job.id,
                run_id=job.run_id,
                track_id=track.track_id,
                purpose="magic_clean_input",
            )
            try:
                if not await self._set_stage(db, job, track_job, "enhancing"):
                    return
                out = await self._enhancer.enhance(
                    input_path=local_path,
                    track_id=track.track_id,
                    job_id=f"{job.id}-{track.track_id}",
                    ai_job_id=job.id,
                    ai_run_id=job.run_id,
                )
                enhanced[track.track_id] = {
                    "enhanced_url": out.enhanced_url,
                    "b2_key": out.b2_key,
                    "quality_score": out.quality_score,
                    "snr_db": out.snr_db,
                }
                speed_source = out.local_path
                cleanup_paths.append(speed_source)
            finally:
                drop_temp_standalone(local_path)

        content_description = None
        discovery_dict = None
        speed_layers: list[dict] = []
        playback_speeds_applied: list[float] = []
        try:
            opts = self._job_options_dict(job)
            llm_speeds = await self._llm_instruction_speeds(job)
            job_sm = self._coerce_job_speed_multipliers(opts.get("speed_multipliers"))
            default_speeds = parse_speed_multiplier_csv(settings.PIPELINE_SPEED_MULTIPLIERS)
            playback_speeds_applied = merge_speed_multipliers(default_speeds, job_sm, llm_speeds)

            tx = effective_transcript_text(track.transcription) if track.transcription else ""
            if tx.strip():
                if await self._set_stage(db, job, track_job, "discovering"):
                    discovery_dict, content_description = await self._run_discovery(
                        track, tx, None, partial_transcript=True
                    )
                    track_job.discovery_json = discovery_dict
                    track_job.updated_at = datetime.utcnow()
                    await commit_with_retry(db)

            if speed_source and os.path.isfile(speed_source) and playback_speeds_applied:
                speed_layers = await self._upload_speed_layers(
                    track_id=track.track_id,
                    job=job,
                    source_path=speed_source,
                    speed_list=playback_speeds_applied,
                )
        finally:
            for p in cleanup_paths:
                if p and os.path.isfile(p):
                    drop_temp_standalone(p)

        result = {
            "job_id": job.id,
            "run_id": job.run_id,
            "job_type": "magic_clean",
            "track_id": track.track_id,
            "enhancement": enhanced.get(track.track_id, {}),
            "discovery": discovery_dict,
            "content_description": content_description,
            "speed_layers": speed_layers,
            "playback_speeds_applied": playback_speeds_applied,
        }
        if b2_deleted:
            result["b2_cleanup"] = {"deleted_keys": b2_deleted}
        completed = await self._complete(db, job, track_job, result)
        if not completed:
            return

    async def _process(self, job_id: str, run_id: str):
        callback_job_id = None
        failed_sse: dict | None = None
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
                source_path = await download_audio(
                    audio_url,
                    suffix=".wav",
                    db=db,
                    job_id=job.id,
                    run_id=job.run_id,
                    track_id=track_id or None,
                    purpose="reconstruct_source",
                )
                try:
                    rebuilt_audio = await self._synthesizer.reconstruct_segments(
                        original_audio_path=source_path,
                        track_id=track_id or "unknown-track",
                        changes=changes,
                        same_speaker=same_speaker,
                    )
                finally:
                    drop_temp_standalone(source_path)
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

            if job.job_type == "audio_tag":
                if not await self._set_stage(db, job, track_job, "audio_tagging"):
                    return
                opts = self._job_options_dict(job)
                media_file_id = opts.get("media_file_id") or self._coerce_transcript_text(job.track_id or "")
                audio_url = self._coerce_transcript_text(
                    job.input_url or opts.get("audio_url") or ""
                )

                if not audio_url:
                    raise ValueError("audio_tag requires audio_url")

                source_path = await download_audio(
                    audio_url,
                    suffix=".wav",
                    db=db,
                    job_id=job.id,
                    run_id=job.run_id,
                    track_id=media_file_id or None,
                    purpose="audio_tag_source",
                )
                try:
                    with open(source_path, "rb") as f:
                        audio_bytes = f.read()
                    if not audio_bytes:
                        raise ValueError("audio_tag source file is empty")
                    raw_transcript = await self._transcriber.transcribe(
                        audio_bytes,
                        job_id=job.id,
                        run_id=job.run_id,
                        track_id=media_file_id or "",
                        short_utterance=True,
                    )
                    transcript_text = self._coerce_transcript_text((raw_transcript or {}).get("transcript", ""))
                    extracted_tags = self._extract_spoken_tags(transcript_text)
                    if not extracted_tags:
                        print(
                            f"[AUDIO_TAG] No speech detected job={job.id} "
                            f"silent={bool((raw_transcript or {}).get('silent'))}"
                        )
                finally:
                    drop_temp_standalone(source_path)

                tag_type = opts.get("type", "track")
                result_field = self._audio_tag_result_field(str(tag_type) if tag_type else "track")
                result: dict = {
                    "job_id": job.id,
                    "job_type": "audio_tag",
                    "media_file_id": media_file_id,
                    "type": tag_type,
                    result_field: extracted_tags,
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
                original_path = await download_audio(
                    track.audio_url,
                    suffix=".wav",
                    db=db,
                    job_id=job.id,
                    run_id=job.run_id,
                    track_id=track.track_id,
                    purpose="rebuild_source",
                )
                try:
                    rebuilt_audio = await self._synthesizer.rebuild_track_audio(
                        original_audio_path=original_path,
                        edited_transcript=job.edited_transcript,
                        track_id=track.track_id,
                        job_id=f"{job.id}-{job.run_id}",
                        original_transcript=self._coerce_transcript_text(track.transcription),
                    )
                finally:
                    drop_temp_standalone(original_path)
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
                if not transcript_text:
                    if not await self._set_stage(db, job, track_job, "transcribing"):
                        return
                    tmp_path = await download_audio(
                        track.audio_url,
                        suffix=".wav",
                        db=db,
                        job_id=job.id,
                        run_id=job.run_id,
                        track_id=track.track_id,
                        purpose="categorization_source",
                    )
                    with open(tmp_path, "rb") as f:
                        audio_bytes = f.read()
                    transcript_data = await self._transcriber.transcribe(
                        audio_bytes,
                        job_id=job.id,
                        run_id=job.run_id,
                        track_id=track.track_id,
                    )
                    transcript_text = self._coerce_transcript_text((transcript_data or {}).get("transcript", ""))
                    segments = self._coerce_segments((transcript_data or {}).get("segments", []))
                else:
                    segments = []
                    transcript_data = {
                        "transcript": transcript_text,
                        "segments": segments,
                        "language": "en",
                        "confidence": 1.0,
                        "edited": bool(job.edited_transcript),
                    }
            else:
                reused_transcript = (
                    effective_transcript_text(track.transcription) if track.transcription else ""
                )
                if reused_transcript and job.job_type != "transcription":
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
                    tmp_path = await download_audio(
                        track.audio_url,
                        suffix=".wav",
                        db=db,
                        job_id=job.id,
                        run_id=job.run_id,
                        track_id=track.track_id,
                        purpose="pipeline_source",
                    )
                    with open(tmp_path, "rb") as f:
                        audio_bytes = f.read()
                    transcript_data = await self._transcriber.transcribe(
                        audio_bytes,
                        job_id=job.id,
                        run_id=job.run_id,
                        track_id=track.track_id,
                    )
                    transcript_text = self._coerce_transcript_text((transcript_data or {}).get("transcript", ""))
                    segments = self._coerce_segments((transcript_data or {}).get("segments", []))

            track_job.transcript = transcript_text or None
            track_job.updated_at = datetime.utcnow()
            await commit_with_retry(db)

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
                await commit_with_retry(db)
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
            await commit_with_retry(db)

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
                await commit_with_retry(db)

            discovery_dict = None
            content_description = None
            if (
                transcript_text
                and not moderation.get("flagged")
                and job.job_type in ("pipeline", "categorization", "rebuild")
            ):
                if await self._set_stage(db, job, track_job, "discovering"):
                    discovery_dict, content_description = await self._run_discovery(
                        track, transcript_text, categorization
                    )
                    track_job.discovery_json = discovery_dict
                    track_job.updated_at = datetime.utcnow()
                    await commit_with_retry(db)

            speed_layers: list[dict] = []
            playback_speeds_applied: list[float] = []
            b2_deleted_keys: list[str] = []
            compressed_audio = None
            if job.job_type == "pipeline":
                wav_for_mp3 = None
                own_wav = False
                if tmp_path and os.path.isfile(tmp_path):
                    wav_for_mp3 = tmp_path
                else:
                    wav_for_mp3 = await download_audio(
                        track.audio_url,
                        suffix=".wav",
                        db=db,
                        job_id=job.id,
                        run_id=job.run_id,
                        track_id=track.track_id,
                        purpose="compress_source",
                    )
                    own_wav = True
                try:
                    b2_deleted_keys = cleanup_track_ai_b2_assets(storage, track, include_enhanced=False)
                    opts = self._job_options_dict(job)
                    llm_speeds = await self._llm_instruction_speeds(job)
                    job_sm = self._coerce_job_speed_multipliers(opts.get("speed_multipliers"))
                    default_speeds = parse_speed_multiplier_csv(settings.PIPELINE_SPEED_MULTIPLIERS)
                    playback_speeds_applied = merge_speed_multipliers(default_speeds, job_sm, llm_speeds)

                    loop = asyncio.get_event_loop()
                    bitrate = settings.PIPELINE_MP3_BITRATE_KBPS
                    mp3_local = await loop.run_in_executor(
                        None,
                        partial(
                            convert_wav_file_to_mp3,
                            wav_for_mp3,
                            bitrate,
                            job_id=job.id,
                            run_id=job.run_id,
                            track_id=track.track_id,
                            purpose="pipeline_mp3",
                        ),
                    )
                    try:
                        b2_key = f"{settings.B2_PIPELINE_MP3_PREFIX}{track.track_id}/{job.id}-{job.run_id}.mp3"
                        url = await loop.run_in_executor(
                            None,
                            partial(storage.upload_file, mp3_local, b2_key, "audio/mpeg"),
                        )
                        compressed_audio = {
                            "audio_url": url,
                            "b2_key": b2_key,
                            "audio_format": "mp3",
                        }
                    finally:
                        drop_temp_standalone(mp3_local)

                    if wav_for_mp3 and os.path.isfile(wav_for_mp3) and playback_speeds_applied:
                        speed_layers = await self._upload_speed_layers(
                            track_id=track.track_id,
                            job=job,
                            source_path=wav_for_mp3,
                            speed_list=playback_speeds_applied,
                            bitrate_kbps=bitrate,
                        )
                except Exception as exc:
                    print(f"[WORKER] Pipeline compressed audio / speed upload skipped: {exc}")
                finally:
                    if own_wav and wav_for_mp3:
                        drop_temp_standalone(wav_for_mp3)

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
                "compressed_audio": compressed_audio if job.job_type == "pipeline" else None,
                "rebuilt_audio": {
                    "audio_url": rebuilt_audio.audio_url,
                    "b2_key": rebuilt_audio.b2_key,
                    "duration": rebuilt_audio.duration,
                } if job.job_type == "rebuild" else None,
            }
            if discovery_dict is not None:
                result["discovery"] = discovery_dict
            if content_description:
                result["content_description"] = content_description
            if job.job_type == "pipeline":
                result["speed_layers"] = speed_layers
                result["playback_speeds_applied"] = playback_speeds_applied
                if b2_deleted_keys:
                    result["b2_cleanup"] = {"deleted_keys": b2_deleted_keys}
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
                non_retryable = isinstance(e, (ValueError, TypeError, AttributeError))
                if not non_retryable and job.attempts < settings.JOB_MAX_RETRIES:
                    job.status = "queued"
                    job.current_stage = None
                    job.attempts += 1
                    if track_job:
                        track_job.status = "queued"
                        track_job.current_stage = None
                        track_job.error = str(e)[:500]
                        track_job.attempts += 1
                        track_job.updated_at = datetime.utcnow()
                    await commit_with_retry(fail_db)
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
                await commit_with_retry(fail_db)
                try:
                    cleanup_job_temp(fail_db, job.id, job.run_id)
                    await commit_with_retry(fail_db)
                except Exception as exc:
                    print(f"[TEMP] cleanup_job_temp on failure failed for {job.id}: {exc}")
                if job.callback_url:
                    callback_job_id = job.id
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
            if failed_sse:
                try:
                    await manager.broadcast(failed_sse["job_id"], failed_sse)
                except Exception:
                    pass
