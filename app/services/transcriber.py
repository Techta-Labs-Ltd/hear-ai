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
    _MIN_WORD_CONFIDENCE = 0.40
    _MIN_TRANSCRIPT_CONFIDENCE = 0.45
    _MIN_REAL_WORDS = 3

    def _run(self, path: str) -> dict:
        _silent = {
            "transcript": "", "segments": [], "language": None,
            "language_probability": 0.0, "duration": 0.0,
            "confidence": 0.0, "silent": True,
        }
        try:
            segments_gen, info = self._model.transcribe(
                path,
                beam_size=5,
                language=None,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=400, speech_pad_ms=200),
                word_timestamps=True,
                condition_on_previous_text=False,
            )
        except ValueError:
            return _silent

        segments = []
        full_text_parts = []
        total_conf = 0.0
        word_count = 0

        for seg in segments_gen:
            # Skip segments Whisper itself flagged as likely no-speech
            if getattr(seg, "no_speech_prob", 0) > 0.6:
                continue

            text = seg.text.strip()
            if not text or all(c in " \t\n.,-!?;:" for c in text):
                continue

            words = []
            for w in (seg.words or []):
                word_text = w.word.strip()
                if not word_text:
                    continue
                # Drop low-confidence words — hallucinated words score poorly
                if w.probability < self._MIN_WORD_CONFIDENCE:
                    continue
                words.append({"word": w.word, "start": w.start, "end": w.end, "prob": w.probability})
                total_conf += w.probability
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

        # Check for known Whisper hallucination phrases
        normalized = transcript.strip().lower().rstrip(".,!?")
        if normalized in self._HALLUCINATION_PHRASES:
            print(f"[TRANSCRIBER] Hallucination detected: {transcript!r} — marking silent")
            return _silent

        # Discard if overall confidence is too low (noise/fan produces low-conf words)
        if confidence < self._MIN_TRANSCRIPT_CONFIDENCE:
            print(f"[TRANSCRIBER] Low confidence transcript ({confidence:.2f}): {transcript!r} — marking silent")
            return _silent

        # Discard if too few real words survived the confidence filter
        if word_count < self._MIN_REAL_WORDS:
            print(f"[TRANSCRIBER] Too few confident words ({word_count}): {transcript!r} — marking silent")
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
                    beam_size=5,
                    vad_filter=True,
                    word_timestamps=True,
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
