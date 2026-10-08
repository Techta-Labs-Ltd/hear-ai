import numpy as np
import pytest
import soundfile as sf

from hear.execution.native import NativeExecutor
from hear.services.categorization.audio_content import AudioContentService
from hear.services.categorization.service import CategorizationService

# Per-clip AudioSet scores as measured on real recordings (see AudioContentService).
SONG = {"Music": 0.93, "Singing": 0.2, "Speech": 0.01, "Silence": 0.0}
INSTRUMENTAL = {"Music": 0.85, "Singing": 0.0, "Speech": 0.0, "Silence": 0.0}
SPEECH = {"Music": 0.0, "Singing": 0.0, "Speech": 0.88, "Silence": 0.0}
SPEECH_OVER_BED = {"Music": 0.56, "Singing": 0.0, "Speech": 0.77, "Silence": 0.0}


class Tagger:
    def __init__(self, scores):
        self.scores = scores
        self.clips = []

    def tag_sync(self, clips, labels):
        self.clips.extend(clips)
        return [self.scores[i % len(self.scores)] for i in range(len(clips))]


def _tone(path, seconds, amplitude=0.2):
    t = np.arange(int(16000 * seconds)) / 16000
    sf.write(path, (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 16000)
    return str(path)


async def _analyse(tagger, path, seconds, *, has_speech):
    native = NativeExecutor("audio-content-test")
    try:
        return await AudioContentService(tagger, native).analyse(
            path, seconds, has_speech=has_speech
        )
    finally:
        await native.close()


@pytest.mark.anyio
async def test_an_instrumental_without_words_is_music(tmp_path):
    tagger = Tagger([INSTRUMENTAL])
    result = await _analyse(tagger, _tone(tmp_path / "a.wav", 95), 95, has_speech=False)
    assert result["kind"] == "music" and result["music"] is True
    assert result["music_share"] == 1.0
    # A short recording is tiled with ten-second clips.
    assert result["clips"] == 10 and all(len(clip) == 160000 for clip in tagger.clips)


def test_long_recordings_are_sampled_with_evenly_spread_clips():
    offsets, length = AudioContentService._offsets(3600.0)
    assert length == 10.0 and len(offsets) == 12
    assert offsets[0] == pytest.approx(149.58, abs=0.01) and offsets[-1] < 3590


@pytest.mark.anyio
async def test_a_song_with_transcribed_lyrics_is_still_music(tmp_path):
    result = await _analyse(Tagger([SONG]), _tone(tmp_path / "a.wav", 60), 60, has_speech=True)
    assert result["music"] is True and result["singing"] is False


@pytest.mark.anyio
async def test_speech_over_a_music_bed_is_not_a_song(tmp_path):
    result = await _analyse(
        Tagger([SPEECH_OVER_BED]), _tone(tmp_path / "a.wav", 60), 60, has_speech=True
    )
    assert result["music"] is False and result["kind"] == "speech"


@pytest.mark.anyio
async def test_a_talk_with_a_music_intro_is_speech_with_music(tmp_path):
    scores = [INSTRUMENTAL, SPEECH, SPEECH, SPEECH, SPEECH, SPEECH]
    result = await _analyse(Tagger(scores), _tone(tmp_path / "a.wav", 180), 180, has_speech=True)
    assert result["music"] is False and result["kind"] == "speech_with_music"


@pytest.mark.anyio
async def test_quiet_clips_are_not_sent_to_the_tagger(tmp_path):
    tagger = Tagger([INSTRUMENTAL])
    path = _tone(tmp_path / "a.wav", 40, amplitude=1e-5)
    result = await _analyse(tagger, path, 40, has_speech=False)
    assert result["kind"] == "silence" and tagger.clips == []


@pytest.mark.anyio
async def test_a_tagger_failure_never_fails_the_job(tmp_path):
    class Broken:
        def tag_sync(self, clips, labels):
            raise RuntimeError("cuda error")

    result = await _analyse(Broken(), _tone(tmp_path / "a.wav", 20), 20, has_speech=False)
    assert result["kind"] == "unknown" and result["music"] is False


class Catalog:
    def __init__(self, categories):
        self._categories = categories

    def flat_catalog_categories(self):
        return self._categories


def test_music_without_words_is_categorised_as_a_song():
    result = CategorizationService(categories=Catalog(["News", "Sport"])).with_song(
        None, confidence=1.0
    )
    assert result["categories"] == ["Song"]
    assert result["tags"] == ["#song", "#music"]
    assert result["confidence_scores"] == {"Song": 1.0}
    assert result["categorizer_mode"] == "audio"


def test_song_is_added_first_to_lyric_categories_in_the_catalog_spelling():
    existing = {
        "tags": ["#love", "#music"],
        "categories": ["Relationships", "songs"],
        "confidence_scores": {"Relationships": 0.7},
        "sentiment": "positive",
    }
    result = CategorizationService(categories=Catalog(["Songs", "News"])).with_song(
        existing, confidence=0.9, max_tags=8
    )
    assert result["categories"] == ["Songs", "Relationships"]
    assert result["tags"] == ["#song", "#music", "#love"]
    assert result["confidence_scores"] == {"Relationships": 0.7, "Songs": 0.9}
    assert result["sentiment"] == "positive"
