from __future__ import annotations

import gc
import json
import os
import tempfile
from functools import partial
from typing import Any

import numpy as np
import torch
import whisperx
from whisperx.asr_qwen import load_model as load_qwen_asr_model

from hear.config import settings
from hear.core.blocking import AsyncCompletion, NativeWorker
from hear.core.hear_temp import TempWorkspace
from hear.utils.transcription_chunks import (
    adaptive_batch_size,
    append_shifted_result,
    finalize_combined_result,
    iter_audio_chunks,
)


class QwenAsrEngine:
    def __init__(self) -> None:
        self._worker = NativeWorker("qwen-asr")
        self._healthy = True
        self._asr = load_qwen_asr_model(
            settings.QWEN_ASR_MODEL_PATH,
            device="cuda",
            language="en",
            download_root=settings.MODEL_CACHE_DIR,
            local_files_only=True,
            vad_options={
                "vad_onset": settings.WHISPER_VAD_ONSET,
                "vad_offset": settings.WHISPER_VAD_OFFSET,
            },
            qwen_dtype=settings.QWEN_ASR_DTYPE,
            qwen_device_map=settings.QWEN_ASR_DEVICE_MAP,
            qwen_forced_aligner=settings.ALIGNER_MODEL_PATH,
            max_inference_batch_size=settings.WHISPER_BATCH_SIZE,
        )
        wrapper = getattr(self._asr, "model", None)
        backend = getattr(wrapper, "model", None)
        thinker = getattr(backend, "thinker", None)
        generation_config = getattr(thinker, "generation_config", None)
        if generation_config is not None and generation_config.pad_token_id is None:
            generation_config.pad_token_id = 151643

    @property
    def ready(self) -> bool:
        return self._healthy

    async def transcribe_window(self, samples: np.ndarray, batch_size: int, language: str) -> dict:
        if samples.ndim != 1 or not 0 < samples.size <= 600 * 16000:
            raise ValueError("invalid_transcription_window")
        if not np.isfinite(samples).all():
            raise ValueError("invalid_transcription_window")
        if not 1 <= batch_size <= settings.WHISPER_BATCH_SIZE:
            raise ValueError("invalid_transcription_batch")
        return await self._worker.run(self._transcribe_window, samples, batch_size, language)

    async def transcribe_bytes(self, audio_bytes: bytes, batch_size: int | None = None) -> dict:
        if len(audio_bytes) > 16 * 1024 * 1024:
            raise ValueError("reference_audio_too_large")
        raw = await self._worker.run(
            self._transcribe_bytes,
            audio_bytes,
            batch_size or settings.WHISPER_BATCH_SIZE,
        )
        return json.loads(raw)

    def _transcribe_window(self, samples: np.ndarray, batch_size: int, language: str) -> dict:
        try:
            with torch.inference_mode():
                result = self._asr.transcribe(samples, batch_size=batch_size, language=language)
            if not isinstance(result, dict) or not isinstance(result.get("segments"), list):
                raise RuntimeError("invalid_transcription_window_result")
            return result
        except RuntimeError as exc:
            if "cuda" in str(exc).lower() or "out of memory" in str(exc).lower():
                self._healthy = False
            raise

    def _transcribe_bytes(self, audio_bytes: bytes, batch_size: int) -> str:
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
            dir=TempWorkspace.hear_temp_directory(),
        ) as stream:
            stream.write(audio_bytes)
            path = stream.name
        try:
            audio = whisperx.load_audio(path)
            duration = len(audio) / 16000
            effective_batch = adaptive_batch_size(
                duration,
                batch_size,
                settings.WHISPER_LONG_AUDIO_BATCH_SIZE,
            )
            combined: dict[str, Any] = {
                "segments": [],
                "language": "en",
                "audio_duration": duration,
            }
            for offset, chunk in iter_audio_chunks(
                audio,
                sample_rate=16000,
                chunk_seconds=settings.WHISPER_CHUNK_SECONDS,
            ):
                with torch.inference_mode():
                    result = self._asr.transcribe(
                        chunk,
                        batch_size=effective_batch,
                        language="en",
                    )
                append_shifted_result(combined, result, offset_seconds=offset)
            return json.dumps(finalize_combined_result(combined))
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
            gc.collect()

    async def close(self) -> None:
        await self._worker.close()
        await AsyncCompletion.run_blocking_to_completion(partial(self._release))

    def _release(self) -> None:
        if hasattr(self, "_asr"):
            del self._asr
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
