"""Inputs libsndfile cannot open are decoded with ffmpeg; readable files never touch it."""

import asyncio
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from hear.services.magic_clean.contracts import CleanExecutionError
from hear.services.transcription.service import TranscriptionService

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")

SECONDS = 4.0


def encode(tmp_path, name: str, codec: list[str]) -> str:
    wav = tmp_path / "tone.wav"
    t = np.arange(int(48000 * SECONDS)) / 48000
    sf.write(wav, 0.3 * np.sin(2 * np.pi * 440 * t), 48000)
    target = tmp_path / name
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), *codec, str(target)], check=True)
    return str(target)


def padded(tmp_path, source: str) -> str:
    """Legacy TN uploads: 417 zero bytes before an otherwise normal MP3."""
    target = tmp_path / "padded.mp3"
    target.write_bytes(b"\x00" * 417 + open(source, "rb").read())
    return str(target)


class CapturingModel:
    def __init__(self) -> None:
        self.samples: list[np.ndarray] = []

    async def transcribe_window(self, samples, batch_size, language, segments=None):
        self.samples.append(samples)
        return {"segments": [{"text": "tone", "start": 0.0, "end": len(samples) / 16000}]}


def transcribe(path: str) -> tuple[dict, CapturingModel]:
    model = CapturingModel()
    result = asyncio.run(TranscriptionService(model, chunk_seconds=600).transcribe_file(path))
    return result, model


def test_readable_mp3_takes_the_libsndfile_path_without_ffmpeg(tmp_path, monkeypatch):
    mp3 = encode(tmp_path, "normal.mp3", ["-c:a", "libmp3lame", "-b:a", "96k"])
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a) or SimpleNamespace())
    result, model = transcribe(mp3)
    assert calls == []
    assert "ffmpeg_fallback_seconds" not in result["performance"]
    assert result["audio_duration"] == pytest.approx(SECONDS, abs=0.1)


def test_ffmpeg_fallback_decodes_zero_padded_legacy_mp3(tmp_path, monkeypatch):
    # Some legacy TN uploads are rejected by the workers' libsndfile ("Format not
    # recognised"); force the fallback so the test covers ffmpeg on padded input.
    source = padded(tmp_path, encode(tmp_path, "normal.mp3", ["-c:a", "libmp3lame", "-b:a", "96k"]))
    monkeypatch.setattr(TranscriptionService, "_libsndfile_can_read", staticmethod(lambda path: False))
    result, model = transcribe(source)
    assert "ffmpeg_fallback_seconds" in result["performance"]
    assert result["audio_duration"] == pytest.approx(SECONDS, abs=0.1)
    audio = np.concatenate(model.samples)
    assert len(audio) == pytest.approx(SECONDS * 16000, abs=1600)
    assert 0.15 < float(np.sqrt(np.mean(audio**2))) < 0.3  # the tone, not silence or noise


def test_m4a_upload_falls_back_to_ffmpeg(tmp_path):
    source = encode(tmp_path, "phone.m4a", ["-c:a", "aac", "-b:a", "96k"])
    result, model = transcribe(source)
    assert "ffmpeg_fallback_seconds" in result["performance"]
    assert result["audio_duration"] == pytest.approx(SECONDS, abs=0.1)


def test_undecodable_input_is_a_final_invalid_audio_error(tmp_path):
    junk = tmp_path / "junk.mp3"
    junk.write_bytes(b"\x00" * 4096)
    with pytest.raises(CleanExecutionError) as error:
        transcribe(str(junk))
    assert error.value.code.value == "invalid_audio" and not error.value.retryable
