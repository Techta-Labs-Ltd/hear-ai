from __future__ import annotations

import gc
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import whisperx
from whisperx.asr_qwen import load_model as load_qwen_asr_model

from hear.execution.native import NativeExecutor
from hear.utils.transcription_chunks import (
    adaptive_batch_size,
    append_shifted_result,
    finalize_combined_result,
    iter_audio_chunks,
)


class QwenAsrEngine:
    def __init__(
        self,
        *,
        model_path: Path,
        aligner_path: Path,
        cache_dir: Path,
        temp_dir: Path,
        dtype: str,
        device_map: str,
        vad_onset: float,
        vad_offset: float,
        max_batch_size: int,
        long_audio_batch_size: int,
        chunk_seconds: int,
    ) -> None:
        self._worker = NativeExecutor("qwen-asr")
        self._temp_dir = temp_dir
        self._max_batch_size = max_batch_size
        self._long_audio_batch_size = long_audio_batch_size
        self._chunk_seconds = chunk_seconds
        self._cuda_healthy = True
        self._asr = load_qwen_asr_model(
            str(model_path),
            device="cuda",
            language="en",
            download_root=str(cache_dir),
            local_files_only=True,
            vad_options={
                "vad_onset": vad_onset,
                "vad_offset": vad_offset,
            },
            qwen_dtype=dtype,
            qwen_device_map=device_map,
            qwen_forced_aligner=str(aligner_path),
            max_inference_batch_size=max_batch_size,
        )
        wrapper = getattr(self._asr, "model", None)
        model = getattr(wrapper, "model", None)
        thinker = getattr(model, "thinker", None)
        generation = getattr(thinker, "generation_config", None)
        if generation is not None and generation.pad_token_id is None:
            generation.pad_token_id = 151643

    async def transcribe_window(
        self,
        samples: np.ndarray,
        batch_size: int,
        language: str,
    ) -> dict:
        if not isinstance(samples, np.ndarray) or samples.ndim != 1:
            raise ValueError("invalid_transcription_window")
        if not 0 < samples.size <= 600 * 16000:
            raise ValueError("invalid_transcription_window")
        if not np.isfinite(samples).all():
            raise ValueError("invalid_transcription_window")
        if not 1 <= batch_size <= self._max_batch_size:
            raise ValueError("invalid_transcription_batch")
        return await self._worker.run(
            self._transcribe_window,
            samples,
            batch_size,
            language,
        )

    async def transcribe(self, audio_bytes: bytes, batch_size: int) -> dict:
        if len(audio_bytes) > 16 * 1024 * 1024:
            raise ValueError("reference_audio_too_large")
        return await self._worker.run(
            self._transcribe_bytes,
            audio_bytes,
            batch_size,
        )

    def _transcribe_window(
        self,
        samples: np.ndarray,
        batch_size: int,
        language: str,
    ) -> dict:
        try:
            with torch.inference_mode():
                result = self._asr.transcribe(
                    samples,
                    batch_size=batch_size,
                    language=language,
                )
            if not isinstance(result, dict) or not isinstance(result.get("segments"), list):
                raise RuntimeError("invalid_transcription_window_result")
            return result
        except RuntimeError as exc:
            if "cuda" in str(exc).lower() or "out of memory" in str(exc).lower():
                self._cuda_healthy = False
            raise

    def _transcribe_bytes(self, audio_bytes: bytes, batch_size: int) -> dict:
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
            dir=self._temp_dir,
        ) as stream:
            stream.write(audio_bytes)
            path = stream.name
        try:
            audio = whisperx.load_audio(path)
            duration = len(audio) / 16000
            effective_batch = adaptive_batch_size(
                duration,
                batch_size,
                self._long_audio_batch_size,
            )
            combined: dict[str, Any] = {
                "segments": [],
                "language": "en",
                "audio_duration": duration,
            }
            for offset, chunk in iter_audio_chunks(
                audio,
                sample_rate=16000,
                chunk_seconds=self._chunk_seconds,
            ):
                result = self._transcribe_window(
                    chunk,
                    effective_batch,
                    "en",
                )
                append_shifted_result(
                    combined,
                    result,
                    offset_seconds=offset,
                )
            return finalize_combined_result(combined)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def check_health(self) -> None:
        if not self._cuda_healthy:
            raise RuntimeError("qwen_asr_cuda_unhealthy")

    async def close(self) -> None:
        await self._worker.close()
        if hasattr(self, "_asr"):
            del self._asr
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()