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


def test_aligner_placeholder_scores_do_not_fake_confidence_and_zero_words_get_span():
    from hear.services.transcription.service import TranscriptionService

    class Model:
        async def transcribe(self, audio_bytes, batch_size):
            return {
                "segments": [
                    {
                        "id": 0,
                        "start": 0.0,
                        "end": 1.0,
                        "text": "to the",
                        "words": [
                            {"word": "to", "start": 0.40, "end": 0.40, "score": 1.0},
                            {"word": "the", "start": 0.41, "end": 0.80, "score": 1.0},
                        ],
                    }
                ],
                "language": "en",
            }

        async def transcribe_window(self, samples, batch_size, language):
            raise AssertionError("unused")

    import asyncio

    result = asyncio.run(TranscriptionService(Model()).transcribe(b"x"))
    words = result["segments"][0]["words"]
    assert result["confidence"] is None and result["word_confidence_available"] is False
    assert words[0]["prob"] is None
    assert words[0]["start"] == 0.40 and words[0]["end"] == 0.41
    assert words[1]["end"] == 0.80


def _truncated_mp3(tmp_path):
    """An MP3 whose Xing header says 30 s but whose frames stop at about 20 s."""
    full = tmp_path / "full.mp3"
    sf.write(full, np.full(16000 * 30, 0.1, dtype=np.float32), 16000, format="MP3")
    data = full.read_bytes()
    path = tmp_path / "truncated.mp3"
    path.write_bytes(data[: len(data) * 2 // 3])
    assert sf.info(path).frames == 16000 * 30
    return path


def _counting_client(windows):
    async def transcribe_window(audio, batch_size, language, segments=None):
        if len(audio) == 0:
            raise ValueError("invalid_transcription_window")  # as QwenAsrEngine does
        windows.append(len(audio))
        return {"segments": [{"text": "hello", "start": 0, "end": 1, "words": []}]}

    return SimpleNamespace(transcribe_window=transcribe_window)


@pytest.mark.anyio
async def test_header_longer_than_audio_keeps_the_transcript_it_has(tmp_path):
    windows = []
    service = TranscriptionService(_counting_client(windows), chunk_seconds=5)
    result = await service.transcribe_file(str(_truncated_mp3(tmp_path)))
    assert len(windows) == 5
    assert 19 < result["audio_duration"] < 21
    assert 9 < result["performance"]["header_overstated_seconds"] < 11
    assert len(result["segments"]) == 5


@pytest.mark.anyio
async def test_vad_windows_past_the_real_end_are_skipped(tmp_path):
    from hear.services.transcription.vad_pool import WindowResult

    class ReadingPool:
        async def stream(self, path, planned):
            with sf.SoundFile(path) as source:
                for start, frames in planned:
                    source.seek(start)
                    samples = TranscriptionService._read_window(source, frames)
                    yield WindowResult(start, samples, [(0.0, 1.0)], 0.0, 0.0)

    windows = []
    service = TranscriptionService(
        _counting_client(windows), chunk_seconds=5, vad_pool=ReadingPool()
    )
    result = await service.transcribe_file(str(_truncated_mp3(tmp_path)))
    assert len(windows) == 5
    assert 19 < result["audio_duration"] < 21


@pytest.mark.anyio
async def test_damaged_frames_mid_file_resume_from_the_ffmpeg_decode(tmp_path):
    from hear.services.transcription.vad_pool import WindowResult

    source = tmp_path / "damaged.wav"
    sf.write(source, np.full(16000 * 20, 0.1, dtype=np.float32), 16000)

    class DamagedPool:
        """libsndfile gives up at 10 s of the source; ffmpeg's WAV reads cleanly."""

        def __init__(self):
            self.paths = []

        async def stream(self, path, planned):
            self.paths.append(path)
            with sf.SoundFile(path) as audio:
                for start, frames in planned:
                    if not path.endswith(".decoded.wav") and start >= 16000 * 10:
                        raise sf.LibsndfileError(29)  # "Unspecified internal error"
                    audio.seek(start)
                    samples = TranscriptionService._read_window(audio, frames)
                    yield WindowResult(start, samples, [(0.0, 1.0)], 0.0, 0.0)

    windows = []
    pool = DamagedPool()
    service = TranscriptionService(_counting_client(windows), chunk_seconds=5, vad_pool=pool)
    result = await service.transcribe_file(str(source))
    assert pool.paths == [str(source), f"{source}.decoded.wav"]
    assert len(windows) == 4  # two before the damage, two resumed; none repeated
    assert [segment["start"] for segment in result["segments"]] == [0, 5, 10, 15]
    assert result["performance"]["ffmpeg_retry_from_seconds"] == 10
    assert 19.9 < result["audio_duration"] < 20.1


@pytest.mark.anyio
async def test_a_failure_after_the_ffmpeg_decode_is_not_retried_again(tmp_path):
    source = tmp_path / "source.wav"
    sf.write(source, np.full(16000 * 10, 0.1, dtype=np.float32), 16000)

    class BrokenPool:
        async def stream(self, path, planned):
            raise sf.LibsndfileError(29)
            yield  # pragma: no cover

    service = TranscriptionService(_counting_client([]), chunk_seconds=5, vad_pool=BrokenPool())
    with pytest.raises(sf.LibsndfileError):
        await service.transcribe_file(str(source))


def _silent_client():
    async def transcribe_window(audio, batch_size, language, segments=None):
        return {"segments": []}

    return SimpleNamespace(transcribe_window=transcribe_window)


@pytest.mark.anyio
async def test_audible_audio_without_words_is_no_speech_not_silent(tmp_path):
    path = tmp_path / "music.wav"
    t = np.arange(16000 * 12) / 16000
    sf.write(path, (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), 16000)
    result = await TranscriptionService(_silent_client(), chunk_seconds=5).transcribe_file(
        str(path)
    )
    assert result["transcript"] == ""
    assert result["no_speech"] is True
    assert result["silent"] is False
    assert result["audio_level"]["audible_seconds"] == 12
    assert -18 < result["audio_level"]["loudest_second_dbfs"] < -16


@pytest.mark.anyio
async def test_near_silent_audio_is_silent(tmp_path):
    path = tmp_path / "quiet.wav"
    rng = np.random.default_rng(0)
    sf.write(path, (rng.standard_normal(16000 * 12) * 3e-5).astype(np.float32), 16000)
    result = await TranscriptionService(_silent_client(), chunk_seconds=5).transcribe_file(
        str(path)
    )
    assert result["silent"] is True and result["no_speech"] is True
    assert result["audio_level"]["audible_seconds"] == 0
