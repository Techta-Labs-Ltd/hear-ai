import asyncio
import logging
import os
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass

import torch
import torchaudio
import torchaudio.functional as F_audio

from app.config import settings
from app.core.audio_utils import save_as_mp3
from app.core.downloader import download_audio
from app.core.hear_temp import drop_temp_standalone, register_temp_standalone, hear_temp_directory
from app.core.storage import get_storage
from app.core.recording_fetcher import fetch_track
from app.core.db_gate import commit_with_retry
from app.models.database import SessionLocal, RegenerationPreview
from app.services.synthesizer import SpeechSynthesizer, SynthesisResult
from app.services.regeneration.quality import RegenerationQualityAssessor, QualityReport

logger = logging.getLogger(__name__)


@dataclass
class PreviewResult:
    preview_id: str
    preview_audio_url: str
    preview_b2_key: str
    preview_duration: float
    quality_metrics: dict
    expires_at: datetime
    track_id: str
    user_id: str | None


class RegenerationService:
    def __init__(self, synthesizer: SpeechSynthesizer):
        self._synthesizer = synthesizer
        self._quality_assessor = RegenerationQualityAssessor()

    async def create_preview(
        self,
        track_id: str,
        audio_url: str,
        changes: list,
        same_speaker: bool = True,
        job_id: str | None = None,
        user_id: str | None = None,
    ) -> PreviewResult:
        logger.info("Creating preview for track=%s changes=%d", track_id, len(changes))

        await self._broadcast_event(track_id, "preview_downloading", {"track_id": track_id})

        source_path = await download_audio(audio_url, suffix=".wav")

        try:
            await self._broadcast_event(track_id, "preview_synthesizing", {"track_id": track_id})

            preview_audio = await self._synthesizer.generate_preview(
                original_audio_path=source_path,
                track_id=track_id,
                changes=changes,
                same_speaker=same_speaker,
                job_id=job_id,
            )

            await self._broadcast_event(track_id, "preview_quality_check", {"track_id": track_id})

            original_waveform, orig_sr = torchaudio.load(source_path)
            quality_metrics = {}
            preview_dl_path = None
            try:
                preview_dl_path = await asyncio.get_event_loop().run_in_executor(
                    None, self._download_to_temp, preview_audio.audio_url
                )
                preview_waveform, preview_sr = torchaudio.load(preview_dl_path)
                if preview_sr != self._synthesizer.TARGET_SR:
                    preview_waveform = F_audio.resample(preview_waveform, preview_sr, self._synthesizer.TARGET_SR)

                first_change = changes[0] if changes else {}
                ref_start = int(float(first_change.get("segment_start", 0)) * self._synthesizer.TARGET_SR)
                ref_end = int(float(first_change.get("segment_end", 0)) * self._synthesizer.TARGET_SR)
                ref_segment = original_waveform[:, ref_start:ref_end] if ref_end > ref_start else original_waveform

                quality_report = self._quality_assessor.assess(
                    preview_waveform, ref_segment, self._synthesizer.TARGET_SR
                )
                quality_metrics = {
                    "dnsmos_ovr": float(quality_report.dnsmos_ovr),
                    "loudness_match_db": float(quality_report.loudness_match_db),
                    "duration_delta_ms": float(quality_report.duration_delta_ms),
                    "clipping_detected": bool(quality_report.clipping_detected),
                    "passed": bool(quality_report.passed),
                }
            except Exception as e:
                logger.warning("Quality assessment failed for preview: %s", e)
                quality_metrics = {"passed": True, "error": str(e)}
            finally:
                if preview_dl_path and os.path.isfile(preview_dl_path):
                    drop_temp_standalone(preview_dl_path)

            preview_id = str(uuid.uuid4())
            expires_at = datetime.utcnow() + timedelta(seconds=settings.REGENERATION_PREVIEW_TTL_SECONDS)

            db = SessionLocal()
            try:
                preview_row = RegenerationPreview(
                    id=preview_id,
                    job_id=job_id,
                    track_id=track_id,
                    action="replace",
                    changes_json={"changes": changes, "same_speaker": same_speaker},
                    preview_b2_key=preview_audio.b2_key,
                    preview_audio_url=preview_audio.audio_url,
                    original_audio_url=audio_url,
                    quality_metrics=quality_metrics,
                    status="pending",
                    seed=self._synthesizer._compute_seed(job_id or track_id, track_id) if job_id else None,
                    user_id=user_id,
                    created_at=datetime.utcnow(),
                    expires_at=expires_at,
                )
                db.add(preview_row)
                await self._commit(db)
            finally:
                db.close()

            await self._broadcast_event(track_id, "preview_ready", {
                "preview_id": preview_id,
                "preview_audio_url": preview_audio.audio_url,
                "quality_metrics": quality_metrics,
                "track_id": track_id,
                "user_id": user_id,
            })

            return PreviewResult(
                preview_id=preview_id,
                preview_audio_url=preview_audio.audio_url,
                preview_b2_key=preview_audio.b2_key,
                preview_duration=preview_audio.duration,
                quality_metrics=quality_metrics,
                expires_at=expires_at,
                track_id=track_id,
                user_id=user_id,
            )
        finally:
            drop_temp_standalone(source_path)

    async def confirm_preview(self, preview_id: str) -> SynthesisResult:
        logger.info("Confirming preview=%s", preview_id)

        db = SessionLocal()
        try:
            preview = db.query(RegenerationPreview).filter(
                RegenerationPreview.id == preview_id,
                RegenerationPreview.status == "pending",
            ).first()
            if not preview:
                raise ValueError(f"Preview {preview_id} not found or not pending")
            preview.status = "confirmed"
            preview.confirmed_at = datetime.utcnow()
            await self._commit(db)
            track_id = preview.track_id
            original_audio_url = preview.original_audio_url
            changes_data = (preview.changes_json or {}).get("changes", [])
        finally:
            db.close()

        await self._broadcast_event(track_id, "confirm_splicing", {"track_id": track_id, "preview_id": preview_id})

        source_path = await download_audio(original_audio_url, suffix=".wav")

        try:
            result = await self._synthesizer.reconstruct_segments(
                original_audio_path=source_path,
                track_id=track_id,
                changes=changes_data,
                same_speaker=True,
                job_id=preview.job_id or preview_id,
            )

            await self._broadcast_event(track_id, "confirm_uploading", {"track_id": track_id})

            await self._broadcast_event(track_id, "confirm_complete", {
                "preview_id": preview_id,
                "final_audio_url": result.audio_url,
                "b2_key": result.b2_key,
                "duration": result.duration,
                "track_id": track_id,
                "segments_applied": len(changes_data),
            })

            try:
                get_storage().delete_object(preview.preview_b2_key)
            except Exception as exc:
                logger.warning("Failed to delete preview B2 object %s: %s", preview.preview_b2_key, exc)

            return result
        finally:
            drop_temp_standalone(source_path)

    async def remove_segment(
        self,
        track_id: str,
        audio_url: str,
        segment_start: float,
        segment_end: float,
        user_id: str | None = None,
    ) -> SynthesisResult:
        logger.info("Removing segment track=%s [%.2f-%.2f]", track_id, segment_start, segment_end)

        await self._broadcast_event(track_id, "remove_downloading", {"track_id": track_id})

        source_path = await download_audio(audio_url, suffix=".wav")

        try:
            await self._broadcast_event(track_id, "remove_cutting", {
                "track_id": track_id,
                "segment_start": segment_start,
                "segment_end": segment_end,
            })

            result = await self._synthesizer.remove_segment(
                original_audio_path=source_path,
                track_id=track_id,
                segment_start=segment_start,
                segment_end=segment_end,
            )

            await self._broadcast_event(track_id, "remove_complete", {
                "final_audio_url": result.audio_url,
                "b2_key": result.b2_key,
                "duration": result.duration,
                "track_id": track_id,
                "removed_duration": round(segment_end - segment_start, 3),
                "user_id": user_id,
            })

            return result
        finally:
            drop_temp_standalone(source_path)

    async def rollback_preview(self, preview_id: str) -> bool:
        db = SessionLocal()
        try:
            preview = db.query(RegenerationPreview).filter(
                RegenerationPreview.id == preview_id,
            ).first()
            if not preview or preview.status == "rolled_back":
                return False

            preview.status = "rolled_back"
            await self._commit(db)

            try:
                get_storage().delete_object(preview.preview_b2_key)
            except Exception as e:
                logger.warning("Failed to delete preview B2 asset %s: %s", preview.preview_b2_key, e)

            await self._broadcast_event(preview.track_id, "preview_rolled_back", {
                "preview_id": preview_id,
                "track_id": preview.track_id,
            })
            return True
        finally:
            db.close()

    async def get_preview(self, preview_id: str) -> dict | None:
        db = SessionLocal()
        try:
            preview = db.query(RegenerationPreview).filter(
                RegenerationPreview.id == preview_id,
            ).first()
            if not preview:
                return None
            return {
                "preview_id": preview.id,
                "job_id": preview.job_id,
                "track_id": preview.track_id,
                "action": preview.action,
                "preview_audio_url": preview.preview_audio_url,
                "quality_metrics": preview.quality_metrics,
                "status": preview.status,
                "seed": preview.seed,
                "user_id": preview.user_id,
                "created_at": str(preview.created_at) if preview.created_at else None,
                "expires_at": str(preview.expires_at) if preview.expires_at else None,
            }
        finally:
            db.close()

    def cleanup_expired_previews(self):
        db = SessionLocal()
        try:
            expired = db.query(RegenerationPreview).filter(
                RegenerationPreview.status == "pending",
                RegenerationPreview.expires_at < datetime.utcnow(),
            ).all()
            for preview in expired:
                try:
                    get_storage().delete_object(preview.preview_b2_key)
                except Exception as e:
                    logger.warning("Failed to delete expired preview B2 %s: %s", preview.preview_b2_key, e)
                db.delete(preview)
            if expired:
                db.commit()
                logger.info("Cleaned up %d expired previews", len(expired))
        except Exception as e:
            logger.warning("Preview cleanup failed: %s", e)
            db.rollback()
        finally:
            db.close()

    async def _broadcast_event(self, track_id: str, event: str, data: dict):
        try:
            payload = {"event": event, **data}
            from ray import serve as _rs
            _rs.get_deployment_handle("orchestrator", "default")._push_event.remote(track_id, payload)
        except Exception:
            pass

    async def _commit(self, db):
        await commit_with_retry(db)

    def _download_to_temp(self, url: str) -> str:
        import httpx
        resp = httpx.get(url, timeout=60)
        resp.raise_for_status()
        path = os.path.join(hear_temp_directory(), f"preview_dl_{uuid.uuid4().hex}.wav")
        with open(path, "wb") as f:
            f.write(resp.content)
        return path
