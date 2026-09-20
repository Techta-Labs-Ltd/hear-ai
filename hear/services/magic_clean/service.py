from __future__ import annotations

import asyncio
import os
import threading
import time
from functools import partial

import torch

from hear.config import settings
from hear.core.blocking import (
    run_blocking_to_completion as _run_blocking_to_completion,
)
from hear.core.hear_temp import (
    drop_temp_standalone,
    hear_temp_job_dir,
    hear_temp_standalone_dir,
)
from hear.core.storage import B2Storage
from hear.services.magic_clean.lineage import magic_clean_artifact_hashes
from hear.services.magic_clean.models import (
    DEFAULT_STEM_LEVELS,
    ContentMode,
    EnhancementResult,
    StemLevels,
)
from hear.services.magic_clean.pipeline import MagicCleanPipeline
from hear.services.magic_clean.processing.dynamics import DynamicsProcessor
from hear.services.magic_clean.processing.mossformer import MossFormer2Enhancer
from hear.core.noise import NoiseReducer
from hear.services.magic_clean.processing.quality import QualityMetrics
from hear.services.magic_clean.processing.silence import SilenceProcessor
from hear.services.magic_clean.processing.speech import SpeechProcessor
from hear.services.magic_clean.processing.stems import StemSeparator
from hear.services.magic_clean.processing.validation import validate_delivered_audio
from hear.services.magic_clean.streaming import clean_file_streaming


class MagicCleanAudioEnhancer:
    """Own the one validated Magic Clean processing and delivery path."""

    def __init__(self) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._mossformer = MossFormer2Enhancer()
        self._noise = NoiseReducer()
        self._speech = SpeechProcessor()
        self._dynamics = DynamicsProcessor(self._device)
        self._metrics = QualityMetrics()
        self._stem = StemSeparator(self._device)
        self._silence = SilenceProcessor()
        self._pipeline = MagicCleanPipeline(
            mossformer=self._mossformer,
            noise=self._noise,
            speech=self._speech,
            dynamics=self._dynamics,
            stem=self._stem,
            silence=self._silence,
        )
        self._gpu_lock = asyncio.Lock()
        self._loaded = False

    def load(self) -> None:
        self._pipeline.load(
            settings.DEMUCS_MODEL,
            settings.MOSSFORMER_MODEL_PATH,
            settings.DEMUCS_MODEL_PATH,
        )
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    async def enhance(
        self,
        input_path: str,
        track_id: str,
        job_id: str,
        mode: ContentMode = ContentMode.AUTO,
        ai_job_id: str | None = None,
        ai_run_id: str | None = None,
        word_timestamps=None,
        speech: int | None = None,
        music: int | None = None,
        background: int | None = None,
        cut_silence: bool = False,
        storage: B2Storage | None = None,
        expected_source_file_sha256: str | None = None,
        expected_source_pcm_sha256: str | None = None,
    ) -> EnhancementResult:
        del word_timestamps
        if not self._loaded:
            raise RuntimeError("Magic Clean is not loaded")
        if storage is None:
            raise ValueError("missing_storage_context")

        stem_levels = self._stem_levels(speech, music, background)
        selected_mode = ContentMode.SPEECH if mode == ContentMode.AUTO else mode
        started_total = time.perf_counter()

        started = time.perf_counter()
        source_file_sha256, source_pcm_sha256 = await _run_blocking_to_completion(
            partial(magic_clean_artifact_hashes, input_path)
        )
        if (
            expected_source_file_sha256
            and source_file_sha256 != expected_source_file_sha256
        ):
            raise RuntimeError("Magic Clean actor downloaded different source bytes")
        if (
            expected_source_pcm_sha256
            and source_pcm_sha256 != expected_source_pcm_sha256
        ):
            raise RuntimeError("Magic Clean actor downloaded different decoded audio")
        source_hash_seconds = _elapsed(started)

        async with self._gpu_lock:
            job_scoped_output = bool(ai_job_id and ai_run_id)
            output_dir = (
                hear_temp_job_dir(ai_job_id, ai_run_id)
                if ai_job_id and ai_run_id
                else hear_temp_standalone_dir("enhance_output")
            )
            output_path = os.path.join(output_dir, "enhance_output.mp3")
            validation_reference_path = os.path.join(
                output_dir,
                "enhance_reference.wav",
            )
            uploaded_key: str | None = None
            processing_cancelled = threading.Event()
            try:
                clean_result = await _run_blocking_to_completion(
                    partial(
                        clean_file_streaming,
                        self._pipeline,
                        input_path,
                        output_path,
                        device=self._device,
                        mode=selected_mode,
                        levels=stem_levels,
                        cut_silence=cut_silence,
                        chunk_seconds=settings.MAGIC_CLEAN_CHUNK_SECONDS,
                        overlap_seconds=settings.MAGIC_CLEAN_CHUNK_OVERLAP_SECONDS,
                        bitrate_kbps=settings.MAGIC_CLEAN_MP3_BITRATE_KBPS,
                        cancel_event=processing_cancelled,
                        validation_reference_path=validation_reference_path,
                    ),
                    on_cancel=processing_cancelled.set,
                )

                started = time.perf_counter()
                delivered = await _run_blocking_to_completion(
                    partial(
                        validate_delivered_audio,
                        input_path,
                        output_path,
                        cut_silence=cut_silence,
                        retained_reference_path=validation_reference_path,
                        # If any component is intentionally removed, a source
                        # channel containing only that component may correctly
                        # become silent. Full source/channel-retention gates
                        # apply only when every component remains requested.
                        expect_audible=self._preserves_all_source_components(
                            stem_levels
                        ),
                        metrics=self._metrics,
                    )
                )
                validation_seconds = _elapsed(started)

                started = time.perf_counter()
                delivered_file_sha256, delivered_pcm_sha256 = (
                    await _run_blocking_to_completion(
                        partial(magic_clean_artifact_hashes, output_path)
                    )
                )
                hashing_seconds = _elapsed(started)

                uploaded_key = storage.key("enhanced", f"{job_id}.mp3")
                started = time.perf_counter()
                enhanced_url = await _run_blocking_to_completion(
                    partial(
                        storage.upload_file,
                        output_path,
                        uploaded_key,
                        "audio/mpeg",
                        checksum_sha256=delivered_file_sha256,
                    )
                )
                upload_seconds = _elapsed(started)

                stage_times = {
                    **clean_result.stage_times,
                    "hash_source": source_hash_seconds,
                    "validate_delivery": validation_seconds,
                    "hash_artifacts": hashing_seconds,
                    "upload": upload_seconds,
                    "total": _elapsed(started_total),
                }
                return EnhancementResult(
                    b2_key=uploaded_key,
                    enhanced_url=enhanced_url,
                    local_path=output_path,
                    quality_score=delivered.quality_score,
                    snr_db=delivered.snr_db,
                    peak_db=delivered.peak_db,
                    lufs=delivered.lufs,
                    clipping_detected=delivered.clipping_detected,
                    mode_used=selected_mode.value,
                    bucket_name=storage.bucket_name,
                    stage_times=stage_times,
                    source_file_sha256=source_file_sha256,
                    source_pcm_sha256=source_pcm_sha256,
                    delivered_file_sha256=delivered_file_sha256,
                    delivered_pcm_sha256=delivered_pcm_sha256,
                    engine_revision=settings.MAGIC_CLEAN_ENGINE_REVISION,
                )
            except BaseException:
                if uploaded_key is not None:
                    try:
                        await _run_blocking_to_completion(
                            partial(storage.delete_object, uploaded_key)
                        )
                    except BaseException:
                        pass
                raise
            finally:
                drop_temp_standalone(output_path)
                drop_temp_standalone(validation_reference_path)
                if not job_scoped_output:
                    drop_temp_standalone(output_dir)

    @staticmethod
    def _preserves_all_source_components(levels: StemLevels) -> bool:
        return all(
            value > 0
            for value in (levels.speech, levels.music, levels.background)
        )

    @staticmethod
    def _stem_levels(
        speech: int | None,
        music: int | None,
        background: int | None,
    ) -> StemLevels:
        supplied = (speech, music, background)
        if all(value is None for value in supplied):
            return DEFAULT_STEM_LEVELS
        if speech is None or music is None or background is None:
            raise ValueError("speech, music, and background must be supplied together")
        return StemLevels(
            speech=int(speech),
            music=int(music),
            background=int(background),
        )


def _elapsed(started: float) -> float:
    return round(time.perf_counter() - started, 3)
