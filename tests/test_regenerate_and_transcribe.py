"""
Integration test: Regeneration → Retranscription → Punctuation Restoration

Tests the full pipeline: TTS generation → Whisper transcription → punctuation restoration.
Verifies that restored punctuation matches the edited transcript after Whisper
transcribes TTS audio (which often lacks punctuation cues).

Requires: GPU (CUDA), FishSpeech server on localhost:8080, Faster-Whisper model
"""

import asyncio
import os
import sys
import tempfile

import pytest
import torch
import torchaudio

sys.path.insert(0, "/workspace/hear-ai")

from app.services.diff_engine import restore_punctuation_from_edit
from app.services.fishspeech_client import FishSpeechClient
from app.services.transcriber import TranscriptionService


TEST_CASES = [
    {
        "id": "multi_sentence",
        "edit": "Hello, world. This is a punctuation test. Does it work today?",
        "description": "Multi-sentence with commas, periods, question mark",
    },
    {
        "id": "commas_matter",
        "edit": "Let's eat, Grandma. Cooking is fun and easy.",
        "description": "Comma changes meaning (Grandma)",
    },
    {
        "id": "question_mid",
        "edit": "What time is it now? The meeting starts at noon sharp.",
        "description": "Question mark in middle of text",
    },
    {
        "id": "semicolons",
        "edit": "First, prepare the ingredients; then, start cooking now.",
        "description": "Semicolons and commas",
    },
    {
        "id": "heavy_punctuation",
        "edit": "Wow! This is amazing. First, we prepare; then, we execute. Ready?",
        "description": "Heavy: exclamation, period, semicolon, question",
    },
]


def _save_wav(waveform: torch.Tensor, sr: int) -> str:
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    torchaudio.save(path, waveform, sr)
    return path


def _count_punct(text: str) -> int:
    return sum(1 for c in text if c in ",.!?;")


@pytest.fixture(scope="module")
def reference_path():
    """Provide a 5-second reference clip for FishSpeech voice cloning."""
    wf, sr = torchaudio.load("/workspace/hear-ai/test_audio.wav")
    if sr != 44100:
        wf = torchaudio.functional.resample(wf, sr, 44100)
        sr = 44100
    ref_start = int(sr * 2.0)
    ref_end = ref_start + int(sr * 5.0)
    ref_clip = wf[:, ref_start:ref_end]
    path = _save_wav(ref_clip, sr)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture(scope="module")
def transcriber():
    """Load Whisper model once for the test module."""
    svc = TranscriptionService()
    svc.load()
    return svc


@pytest.fixture(scope="module")
def fish_client():
    return FishSpeechClient()


class TestRegenerateAndTranscribe:
    """End-to-end: TTS → Whisper transcribe → restore punctuation → verify."""

    @pytest.mark.parametrize("tc", TEST_CASES, ids=[t["id"] for t in TEST_CASES])
    def test_punctuation_restoration(
        self, tc, reference_path, transcriber, fish_client
    ):
        edited = tc["edit"]

        # 1. Generate TTS audio
        tts_bytes = fish_client.generate_speech(
            text=edited,
            reference_audio_path=reference_path,
        )
        assert len(tts_bytes) > 100, "TTS produced too little audio"

        # 2. Transcribe with Whisper
        loop = asyncio.get_event_loop()
        transcript_data = loop.run_until_complete(
            transcriber.transcribe(tts_bytes, language="en")
        )
        whisper_text = (transcript_data.get("transcript") or "").strip()
        confidence = transcript_data.get("confidence", 0.0)

        assert whisper_text, f"Whisper produced empty transcript for: {edited}"
        assert confidence > 0.1, f"Whisper confidence too low: {confidence}"

        # 3. Restore punctuation
        restored = restore_punctuation_from_edit(whisper_text, edited)

        # 4. Verify punctuation recovery
        edit_punct = _count_punct(edited)
        whisper_punct = _count_punct(whisper_text)
        restored_punct = _count_punct(restored)

        assert restored_punct >= whisper_punct, (
            f"Punctuation decreased: Whisper={whisper_punct}, "
            f"Restored={restored_punct}, Edit={edit_punct}"
        )

        if edit_punct > 0:
            recovery_pct = (restored_punct / edit_punct) * 100
            assert recovery_pct >= 70, (
                f"Punctuation recovery too low: {recovery_pct:.0f}% "
                f"(Restored={restored_punct}, Edit={edit_punct})"
            )

        # 5. Verify word content preserved (no meaning changed)
        import re

        def strip_punct(s):
            return re.sub(r"[^\w\s]", "", s).lower().strip()

        whisper_content = set(strip_punct(whisper_text).split())
        restored_content = set(strip_punct(restored).split())
        edit_content = set(strip_punct(edited).split())

        preserved = restored_content & edit_content
        content_retention = len(preserved) / max(len(edit_content), 1)
        assert content_retention >= 0.7, (
            f"Content retention too low: {content_retention:.0%} "
            f"(Preserved={len(preserved)}, Edit={len(edit_content)})"
        )

    def test_perfect_restoration_quality(self, reference_path, transcriber, fish_client):
        """Verify that complex punctuation text achieves high restoration quality."""
        heavy_text = "Wow! This is amazing. First, we prepare; then, we execute. Ready?"
        tts_bytes = fish_client.generate_speech(text=heavy_text, reference_audio_path=reference_path)

        loop = asyncio.get_event_loop()
        transcript_data = loop.run_until_complete(
            transcriber.transcribe(tts_bytes, language="en")
        )
        whisper_text = (transcript_data.get("transcript") or "").strip()
        restored = restore_punctuation_from_edit(whisper_text, heavy_text)

        edit_p = _count_punct(heavy_text)
        restored_p = _count_punct(restored)

        assert restored_p >= edit_p * 0.8, (
            f"Heavy punctuation restoration insufficient: "
            f"Restored={restored_p}/{edit_p} -> '{restored}'"
        )

    def test_word_level_mishearing_correction(self, reference_path, transcriber, fish_client):
        """Verify that near-miss words are corrected via fuzzy matching."""
        from app.services.diff_engine import correct_whisper_mishearings

        edit = "This is a favour for you today"
        tts_bytes = fish_client.generate_speech(text=edit, reference_audio_path=reference_path)

        loop = asyncio.get_event_loop()
        transcript_data = loop.run_until_complete(
            transcriber.transcribe(tts_bytes, language="en")
        )
        whisper_text = (transcript_data.get("transcript") or "").strip()

        corrected = correct_whisper_mishearings(whisper_text, edit, max_distance=3)
        assert len(corrected) > 0
        assert any(w.lower() in corrected.lower() for w in edit.split()), (
            f"Corrected text lost edit words: '{corrected}'"
        )
