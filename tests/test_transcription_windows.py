from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
import soundfile as sf

from hear.services.transcription.service import TranscriptionService


@pytest.mark.anyio
async def test_file_transcription_uses_bounded_windows_and_global_word_offsets(tmp_path):
    path = tmp_path / "source.wav"
    samples = np.ones((40000, 2), dtype=np.float32) * 0.01
    sf.write(path, samples, 16000)
    windows = []

    async def transcribe_window(audio, batch_size, language):
        windows.append(len(audio))
        return {
            "segments": [
                {
                    "text": "hello",
                    "start": 0,
                    "end": len(audio) / 16000,
                    "words": [
                        {"word": "hello", "start": 0, "end": len(audio) / 16000, "score": 0.9}
                    ],
                }
            ]
        }

    service = TranscriptionService(
        SimpleNamespace(transcribe_window=transcribe_window), chunk_seconds=1
    )
    result = await service.transcribe_file(str(path))
    assert windows == [16000, 16000, 8000]
    assert result["audio_duration"] == 2.5
    assert [segment["start"] for segment in result["segments"]] == [0, 1, 2]
    assert [segment["words"][0]["end"] for segment in result["segments"]] == [1, 2, 2.5]


@pytest.mark.anyio
async def test_file_transcription_resamples_stereo_without_whole_file_transfer(tmp_path):
    path = tmp_path / "source.wav"
    sf.write(path, np.zeros((48000, 2), dtype=np.float32), 48000)
    client = SimpleNamespace(transcribe_window=AsyncMock(return_value={"segments": []}))
    result = await TranscriptionService(client, chunk_seconds=1).transcribe_file(str(path))
    samples = client.transcribe_window.await_args.args[0]
    assert samples.shape == (16000,)
    assert result["silent"] is True
    assert result["audio_duration"] == 1


@pytest.mark.anyio
async def test_invalid_model_response_is_not_silence():
    client = SimpleNamespace(transcribe=AsyncMock(return_value=None))
    with pytest.raises(RuntimeError, match="invalid_transcription_result"):
        await TranscriptionService(client).transcribe(b"reference")


@pytest.mark.anyio
async def test_reference_input_is_bounded_before_model_call():
    client = SimpleNamespace(transcribe=AsyncMock())
    with pytest.raises(ValueError, match="reference_audio_too_large"):
        await TranscriptionService(client).transcribe(bytes(16 * 1024 * 1024 + 1))
    client.transcribe.assert_not_called()
