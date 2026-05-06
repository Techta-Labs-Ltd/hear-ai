import asyncio
import os
import tempfile
from typing import AsyncGenerator

import torch
from faster_whisper import WhisperModel

from app.config import settings

# 
class TranscriptionService:
    def __init__(self):
        self._model = None

    def load(self):
        self._model = WhisperModel(
            settings.WHISPER_MODEL_SIZE,
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16" if torch.cuda.is_available() else "int8",
            num_workers=2,
            download_root=f"{settings.MODEL_CACHE_DIR}/whisper",
        )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    async def transcribe(self, audio_bytes: bytes) -> dict:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._run, tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    _HALLUCINATION_PHRASES = {
        "thank you", "thanks for watching", "thanks for listening",
        "please subscribe", "subscribe", "like and subscribe",
        "you're welcome", "you are welcome", "welcome back",
        "see you next time", "see you later", "bye", "goodbye",
        "good morning", "good evening", "good afternoon", "good night",
        "hello", "hi there", "hey there", "what's up",
        "uh", "um", "hmm", "hm", "ah", "oh",
        ".", "..", "...", "[music]", "[applause]", "[laughter]",
        "[noise]", "[silence]", "[inaudible]", "[blank_audio]",
    }
    _MIN_WORD_CONFIDENCE = 0.20
    _MIN_TRANSCRIPT_CONFIDENCE = 0.30
    _MIN_REAL_WORDS = 2

    def _run(self, path: str) -> dict:
        _silent = {
            "transcript": "", "segments": [], "language": None,
            "language_probability": 0.0, "duration": 0.0,
            "confidence": 0.0, "silent": True,
        }
        strict = self._run_pass(path, relaxed=False)
        if not strict.get("silent") and not settings.WHISPER_DUAL_PASS:
            return strict
        if not strict.get("silent"):
            word_count = len(strict.get("transcript", "").split())
            if word_count >= 40:
                return strict
        relaxed = self._run_pass(path, relaxed=True)
        if relaxed.get("silent"):
            return strict
        strict_words = len(strict.get("transcript", "").split()) if not strict.get("silent") else 0
        relaxed_words = len(relaxed.get("transcript", "").split())
        if relaxed_words > strict_words:
            return relaxed
        return strict if not strict.get("silent") else relaxed

    def _run_pass(self, path: str, relaxed: bool) -> dict:
        _silent = {
            "transcript": "", "segments": [], "language": None,
            "language_probability": 0.0, "duration": 0.0,
            "confidence": 0.0, "silent": True,
        }
        try:
            kwargs = {
                "beam_size": max(1, settings.WHISPER_BEAM_SIZE),
                "language": None,
                "word_timestamps": settings.WHISPER_WORD_TIMESTAMPS,
                "condition_on_previous_text": False,
            }
            if relaxed:
                kwargs["vad_filter"] = False
            else:
                kwargs["vad_filter"] = True
                kwargs["vad_parameters"] = dict(min_silence_duration_ms=400, speech_pad_ms=200)
            segments_gen, info = self._model.transcribe(path, **kwargs)
        except ValueError:
            return _silent

        segments = []
        full_text_parts = []
        total_conf = 0.0
        word_count = 0
        min_word_conf = 0.05 if relaxed else self._MIN_WORD_CONFIDENCE

        for seg in segments_gen:
            if not relaxed and getattr(seg, "no_speech_prob", 0) > 0.6:
                continue
            text = seg.text.strip()
            if not text or all(c in " \t\n.,-!?;:" for c in text):
                continue
            words = []
            for w in (seg.words or []):
                word_text = w.word.strip()
                if not word_text:
                    continue
                if w.probability < min_word_conf:
                    continue
                words.append({"word": w.word, "start": w.start, "end": w.end, "prob": w.probability})
                total_conf += w.probability
                word_count += 1
            if not words and relaxed:
                words = [{"word": text, "start": seg.start, "end": seg.end, "prob": max(float(getattr(seg, "avg_logprob", -1.0)) + 1.0, 0.1)}]
                total_conf += words[0]["prob"]
                word_count += 1
            if not words:
                continue
            segments.append({
                "id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "text": text,
                "words": words,
            })
            full_text_parts.append(text)

        if not full_text_parts:
            return _silent

        transcript = " ".join(full_text_parts)
        confidence = round(total_conf / max(word_count, 1), 4)
        normalized = transcript.strip().lower().rstrip(".,!?")
        if normalized in self._HALLUCINATION_PHRASES:
            return _silent
        if not relaxed and confidence < self._MIN_TRANSCRIPT_CONFIDENCE:
            return _silent
        if not relaxed and word_count < self._MIN_REAL_WORDS:
            return _silent
        return {
            "transcript": transcript,
            "segments": segments,
            "language": info.language,
            "language_probability": round(info.language_probability, 4),
            "duration": info.duration,
            "confidence": confidence,
            "silent": False,
        }

    async def stream(self, audio_bytes: bytes) -> AsyncGenerator[dict, None]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _worker():
            try:
                segments_gen, info = self._model.transcribe(
                    tmp_path,
                    beam_size=max(1, settings.WHISPER_BEAM_SIZE),
                    vad_filter=True,
                    word_timestamps=settings.WHISPER_WORD_TIMESTAMPS,
                    condition_on_previous_text=False,
                )
                for seg in segments_gen:
                    text = seg.text.strip()
                    if not text or all(c in " \t\n.,-!?;:" for c in text):
                        continue
                    words = [
                        {"word": w.word, "start": w.start, "end": w.end, "prob": w.probability}
                        for w in (seg.words or [])
                        if w.word.strip()
                    ]
                    if not words:
                        continue
                    loop.call_soon_threadsafe(queue.put_nowait, {
                        "type": "segment",
                        "id": seg.id,
                        "start": seg.start,
                        "end": seg.end,
                        "text": text,
                        "words": words,
                        "language": info.language,
                    })
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "done", "language": info.language})
            except ValueError:
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "done", "language": None, "silent": True})
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": str(e)})
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        loop.run_in_executor(None, _worker)

        while True:
            item = await queue.get()
            yield item
            if item["type"] in ("done", "error"):
                break
