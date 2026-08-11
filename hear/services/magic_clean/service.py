import asyncio
import os
from functools import partial

import numpy as np
import torch

from hear.config import settings
from hear.core.audio_utils import delivery_bitrate_kbps, probe_audio, save_as_mp3
from hear.core.hear_temp import (
    drop_temp_standalone, hear_temp_job_dir, hear_temp_standalone_dir,
)
from hear.core.storage import B2Storage
from hear.services.magic_clean.models import (
    DEFAULT_STEM_LEVELS,
    ContentMode,
    EnhancementResult,
    StemLevels,
)
from hear.services.magic_clean.pipeline import MagicCleanPipeline
from hear.services.magic_clean.streaming import clean_file_streaming
from hear.services.magic_clean.processing.audio_io import AudioIO
from hear.services.magic_clean.processing.dynamics import DynamicsProcessor
from hear.services.magic_clean.processing.mossformer import MossFormer2Enhancer
from hear.services.magic_clean.processing.noise import NoiseReducer
from hear.services.magic_clean.processing.quality import QualityMetrics
from hear.services.magic_clean.processing.silence import SilenceProcessor
from hear.services.magic_clean.processing.speech import SpeechProcessor
from hear.services.magic_clean.processing.stems import StemSeparator


class MagicCleanAudioEnhancer:
    def __init__(self):
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

    def load(self):
        self._pipeline.load(settings.DEMUCS_MODEL)
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    async def _enhance_long_file(
        self, input_path, track_id, job_id, mode, ai_job_id, ai_run_id,
        stem_levels, cut_silence, storage: B2Storage,
    ):
        loop = asyncio.get_running_loop()
        output_dir = (
            hear_temp_job_dir(ai_job_id, ai_run_id)
            if ai_job_id and ai_run_id
            else hear_temp_standalone_dir("enhance_output")
        )
        out_path = os.path.join(output_dir, "enhance_output.mp3")
        bitrate_kbps = delivery_bitrate_kbps(
            input_path, maximum_kbps=settings.PIPELINE_MP3_BITRATE_KBPS
        )
        clean_result = await loop.run_in_executor(
            None,
            partial(
                clean_file_streaming,
                self._pipeline,
                input_path,
                out_path,
                device=self._device,
                mode=ContentMode.SPEECH if mode == ContentMode.AUTO else mode,
                levels=stem_levels,
                cut_silence=cut_silence,
                chunk_seconds=settings.MAGIC_CLEAN_CHUNK_SECONDS,
                overlap_seconds=settings.MAGIC_CLEAN_CHUNK_OVERLAP_SECONDS,
                bitrate_kbps=bitrate_kbps,
            ),
        )
        b2_key = storage.key("enhanced", f"{job_id}.mp3")
        try:
            url = await loop.run_in_executor(
                None, storage.upload_file, out_path, b2_key, "audio/mpeg"
            )
        except Exception:
            drop_temp_standalone(out_path)
            raise
        return EnhancementResult(
            b2_key=b2_key,
            enhanced_url=url,
            local_path=out_path,
            quality_score=self._metrics.compute_quality_score(
                0.0, False, clean_result.integrated_lufs
            ),
            snr_db=0.0,
            peak_db=round(clean_result.peak_db, 2),
            lufs=round(clean_result.integrated_lufs, 2),
            clipping_detected=False,
            mode_used=(ContentMode.SPEECH if mode == ContentMode.AUTO else mode).value,
            bucket_name=storage.bucket_name,
        )

    async def enhance(self, input_path, track_id, job_id, mode=ContentMode.AUTO,
                      ai_job_id=None, ai_run_id=None, word_timestamps=None,
                      speech=None, music=None, background=None,
                      cut_silence=False, storage: B2Storage | None = None):
        if storage is None:
            raise ValueError("missing_storage_context")
        async with self._gpu_lock:
            loop = asyncio.get_running_loop()
            out_path = None
            try:
                duration = probe_audio(input_path)["duration_seconds"]
                if duration >= settings.MAGIC_CLEAN_STREAMING_THRESHOLD_SECONDS:
                    if mode == ContentMode.AUTO:
                        mode = ContentMode.SPEECH
                    supplied_levels = (speech, music, background)
                    if all(value is None for value in supplied_levels):
                        stem_levels = DEFAULT_STEM_LEVELS
                    elif all(value is not None for value in supplied_levels):
                        stem_levels = StemLevels(
                            speech=int(speech), music=int(music),
                            background=int(background),
                        )
                    else:
                        raise ValueError(
                            "speech, music, and background must be supplied together"
                        )
                    return await self._enhance_long_file(
                        input_path, track_id, job_id, mode, ai_job_id, ai_run_id,
                        stem_levels, cut_silence, storage,
                    )

                waveform, sr = AudioIO.load(input_path)
                clipping = AudioIO.detect_clipping(waveform)
                resampled = AudioIO.resample(
                    AudioIO.to_mono(waveform), sr, AudioIO.TARGET_SR
                )
                raw_target = resampled.clone()
                enhanced = resampled.to(self._device)
                if mode == ContentMode.AUTO:
                    mode = ContentMode.SPEECH

                supplied_levels = (speech, music, background)
                if all(value is None for value in supplied_levels):
                    stem_levels = DEFAULT_STEM_LEVELS
                elif any(value is not None for value in supplied_levels):
                    if not all(value is not None for value in supplied_levels):
                        raise ValueError(
                            "speech, music, and background must be supplied together"
                        )
                    stem_levels = StemLevels(
                        speech=int(speech),
                        music=int(music),
                        background=int(background),
                    )
                else:
                    raise AssertionError("unreachable stem-level state")

                enhanced = await loop.run_in_executor(
                    None,
                    partial(
                        self._pipeline.process_chunked,
                        enhanced,
                        AudioIO.TARGET_SR,
                        mode,
                        stem_levels,
                        cut_silence,
                        chunk_seconds=settings.MAGIC_CLEAN_CHUNK_SECONDS,
                        overlap_seconds=settings.MAGIC_CLEAN_CHUNK_OVERLAP_SECONDS,
                    ),
                )
                enhanced_cpu = enhanced.detach().cpu()
                snr = self._metrics.compute_snr(raw_target, enhanced_cpu)

                lufs = self._metrics.compute_lufs(enhanced_cpu)
                peak_db = 20 * np.log10(enhanced_cpu.abs().max().item() + 1e-8)
                score = self._metrics.compute_quality_score(snr, clipping, lufs)
                bitrate_kbps = delivery_bitrate_kbps(
                    input_path,
                    maximum_kbps=settings.PIPELINE_MP3_BITRATE_KBPS,
                )
                out_path = save_as_mp3(
                    enhanced_cpu, AudioIO.TARGET_SR, job_id=ai_job_id,
                    run_id=ai_run_id, track_id=track_id, purpose="enhance_output",
                    bitrate_kbps=bitrate_kbps,
                )
                b2_key = storage.key("enhanced", f"{job_id}.mp3")
                url = await loop.run_in_executor(
                    None, storage.upload_file, out_path, b2_key, "audio/mpeg"
                )
                return EnhancementResult(
                    b2_key=b2_key, enhanced_url=url, local_path=out_path,
                    quality_score=score, snr_db=round(snr, 2),
                    peak_db=round(peak_db, 2), lufs=round(lufs, 2),
                    clipping_detected=clipping, mode_used=mode.value,
                    bucket_name=storage.bucket_name,
                )
            except Exception:
                if out_path:
                    try:
                        drop_temp_standalone(out_path)
                    except Exception:
                        if os.path.exists(out_path):
                            try:
                                os.unlink(out_path)
                            except OSError:
                                pass
                raise
