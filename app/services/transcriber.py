import asyncio
import tempfile

import torch

from app.config import settings
from app.core.gpu import cuda_inference_lock
from app.core.hear_temp import (
    drop_temp_standalone,
    hear_temp_directory,
    hear_temp_job_dir,
    register_temp_standalone,
)
from app.services.triton_client import get_triton_client


class TranscriptionService:
    def __init__(self):
        self._lock = asyncio.Lock()

    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        job_id: str | None = None,
        run_id: str | None = None,
        track_id: str | None = None,
        short_utterance: bool = False,
        language: str | None = None,
    ) -> dict:
        client = get_triton_client()
        loop = asyncio.get_event_loop()
        async with self._lock:
            result = await loop.run_in_executor(
                None, client.transcribe_sync, audio_bytes, settings.WHISPER_BATCH_SIZE,
            )
        return self._process_result(result, language=language)

    def _process_result(self, result: dict, language: str | None = None) -> dict:
        _silent = {
            "transcript": "", "segments": [], "language": None,
            "language_probability": 0.0, "duration": 0.0,
            "confidence": 0.0, "silent": True,
        }
        if not result:
            return _silent

        segments_list = result.get("segments", [])
        detected_language = result.get("language", language or "en")

        segments = []
        full_text_parts = []
        total_conf = 0.0
        word_count = 0

        for seg in segments_list:
            text = seg.get("text", "").strip()
            if not text:
                continue
            words = []
            for w in (seg.get("words") or []):
                word_text = w.get("word", "").strip()
                if not word_text:
                    continue
                words.append({
                    "word": w["word"], "start": w["start"],
                    "end": w["end"], "prob": w.get("score", 1.0),
                })
                total_conf += w.get("score", 1.0)
                word_count += 1
            if not words:
                avg_logprob = seg.get("avg_logprob", -1.0)
                prob = max(float(avg_logprob) + 1.0, 0.1)
                words.append({
                    "word": text, "start": seg.get("start", 0),
                    "end": seg.get("end", 0), "prob": prob,
                })
                total_conf += prob
                word_count += 1
            segments.append({
                "id": seg.get("id", len(segments)),
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "text": text,
                "words": words,
            })
            full_text_parts.append(text)

        if not full_text_parts:
            return _silent

        transcript = " ".join(full_text_parts)
        confidence = round(total_conf / max(word_count, 1), 4)
        duration = segments[-1]["end"] if segments else 0.0
        return {
            "transcript": transcript,
            "segments": segments,
            "word_segments": result.get("word_segments", []),
            "language": detected_language,
            "language_probability": 1.0,
            "duration": duration,
            "confidence": confidence,
            "silent": False,
        }
