import asyncio
import tempfile
import re

import torch

from hear.config import settings
from hear.core.gpu import cuda_inference_lock
from hear.core.hear_temp import (
    drop_temp_standalone,
    hear_temp_directory,
    hear_temp_job_dir,
)
from hear.services.model_client import get_model_client


_HALLUCINATION_ONLY = {
    "thank you",
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
}


def _normalized_text(value: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", value.lower()).strip()


def _credible_segments(result: dict, *, short_utterance: bool) -> list[dict]:
    segments = list(result.get("segments") or [])
    credible = [
        segment for segment in segments
        if "avg_logprob" not in segment
        or float(segment.get("avg_logprob", -99.0)) >= settings.WHISPER_MIN_AVG_LOGPROB
    ]
    if not credible or short_utterance:
        return credible

    text = " ".join(str(segment.get("text") or "").strip() for segment in credible)
    speech_seconds = sum(
        max(0.0, float(segment.get("end", 0)) - float(segment.get("start", 0)))
        for segment in credible
    )
    audio_seconds = float(result.get("audio_duration") or 0.0)
    if (
        _normalized_text(text) in _HALLUCINATION_ONLY
        and audio_seconds >= 5.0
        and speech_seconds <= 3.0
    ):
        return []
    return credible


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
        client = get_model_client()
        loop = asyncio.get_event_loop()
        async with self._lock:
            result = await loop.run_in_executor(
                None, client.transcribe_sync, audio_bytes, settings.WHISPER_BATCH_SIZE,
            )
        return self._process_result(
            result, language=language, short_utterance=short_utterance
        )

    def _process_result(
        self, result: dict, language: str | None = None,
        short_utterance: bool = False,
    ) -> dict:
        _silent = {
            "transcript": "", "segments": [], "language": None,
            "language_probability": 0.0, "duration": 0.0,
            "confidence": 0.0, "silent": True,
        }
        if not result:
            return _silent

        segments_list = _credible_segments(
            result, short_utterance=short_utterance
        )
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
