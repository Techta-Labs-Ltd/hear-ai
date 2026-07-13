"""Tests for the voice profile store.

No GPU or network needed — tests file I/O operations only.
Uses a temp directory that is cleaned up after each test.
"""

import json
import os
import shutil
import tempfile

import numpy as np
import pytest

from app.services.voice_profile import (
    _METADATA_FILENAME,
    _REFERENCE_AUDIO_FILENAME,
    _REFERENCE_TEXT_FILENAME,
    cleanup_profiles,
    get_or_create_from_track,
    get_reference_path,
    get_reference_text,
    list_profiles,
    save_profile,
)


def _generate_wav_bytes(
    duration_s: float = 1.0,
    sample_rate: int = 24000,
    frequency: float = 440.0,
) -> bytes:
    """Generate a minimal valid WAV file."""
    import io
    import wave

    n_samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    signal = (0.3 * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(signal.tobytes())
    return buf.getvalue()


@pytest.fixture(autouse=True)
def temp_profiles_dir(monkeypatch, tmp_path):
    """Redirect voice profiles to a temp dir for each test."""
    profiles_dir = str(tmp_path / "voice_profiles")
    os.makedirs(profiles_dir, exist_ok=True)

    monkeypatch.setattr("app.services.voice_profile._profiles_root", lambda: profiles_dir)
    monkeypatch.setattr("app.services.voice_profile._profile_dir", lambda user_id: (
        lambda pd=profiles_dir, uid=user_id: (
            d := os.path.join(pd, uid.replace(os.sep, "_").replace("..", "_")),
            os.makedirs(d, exist_ok=True),
            d,
        )[-1]
    )())

    yield profiles_dir

    if os.path.exists(profiles_dir):
        shutil.rmtree(profiles_dir, ignore_errors=True)


class TestSaveProfile:
    def test_creates_files(self, temp_profiles_dir):
        wav = _generate_wav_bytes()
        pdir = save_profile("user-1", wav, "Hello world", language="en")

        assert os.path.isfile(os.path.join(pdir, _REFERENCE_AUDIO_FILENAME))
        assert os.path.isfile(os.path.join(pdir, _REFERENCE_TEXT_FILENAME))
        assert os.path.isfile(os.path.join(pdir, _METADATA_FILENAME))

    def test_writes_transcript(self, temp_profiles_dir):
        wav = _generate_wav_bytes()
        save_profile("user-1", wav, "Hello world test")

        text = get_reference_text("user-1")
        assert text == "Hello world test"

    def test_metadata(self, temp_profiles_dir):
        wav = _generate_wav_bytes(duration_s=2.0, sample_rate=24000)
        save_profile("user-1", wav, "test", language="en")

        pdir = os.path.join(temp_profiles_dir, "user-1")
        with open(os.path.join(pdir, _METADATA_FILENAME)) as f:
            meta = json.load(f)

        assert meta["user_id"] == "user-1"
        assert meta["language"] == "en"
        assert meta["sample_count"] == 1
        assert meta["duration"] > 0

    def test_updates_existing(self, temp_profiles_dir):
        wav1 = _generate_wav_bytes()
        save_profile("user-1", wav1, "first text")

        wav2 = _generate_wav_bytes()
        save_profile("user-1", wav2, "second text")

        text = get_reference_text("user-1")
        assert text == "second text"

        pdir = os.path.join(temp_profiles_dir, "user-1")
        with open(os.path.join(pdir, _METADATA_FILENAME)) as f:
            meta = json.load(f)
        assert meta["sample_count"] == 2


class TestGetReferencePath:
    def test_returns_path_when_exists(self, temp_profiles_dir):
        wav = _generate_wav_bytes()
        save_profile("user-1", wav, "test")

        path = get_reference_path("user-1")
        assert path is not None
        assert path.endswith(".wav")
        assert os.path.isfile(path)

    def test_returns_none_when_missing(self, temp_profiles_dir):
        assert get_reference_path("nonexistent") is None


class TestGetOrCreateFromTrack:
    def test_creates_from_track(self, temp_profiles_dir):
        wav = _generate_wav_bytes(duration_s=3.0)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav)
            tmp_path = tmp.name

        try:
            pdir = get_or_create_from_track("user-1", tmp_path, transcript="track transcript")
            assert os.path.isdir(pdir)

            text = get_reference_text("user-1")
            assert text == "track transcript"
        finally:
            os.unlink(tmp_path)

    def test_reuses_existing(self, temp_profiles_dir):
        wav = _generate_wav_bytes()
        save_profile("user-1", wav, "first")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(_generate_wav_bytes())
            tmp_path = tmp.name

        try:
            pdir = get_or_create_from_track("user-1", tmp_path, transcript="second")
            text = get_reference_text("user-1")
            assert text == "first"
        finally:
            os.unlink(tmp_path)


class TestListProfiles:
    def test_empty(self, temp_profiles_dir):
        assert list_profiles() == []

    def test_lists_multiple(self, temp_profiles_dir):
        save_profile("user-1", _generate_wav_bytes(), "text 1")
        save_profile("user-2", _generate_wav_bytes(), "text 2")

        profiles = list_profiles()
        assert len(profiles) == 2
        ids = {p["user_id"] for p in profiles}
        assert ids == {"user-1", "user-2"}


class TestCleanupProfiles:
    def test_nothing_to_clean(self, temp_profiles_dir):
        save_profile("user-1", _generate_wav_bytes(), "test")
        removed = cleanup_profiles(max_age_hours=168)
        assert removed == 0

    def test_removes_old(self, temp_profiles_dir, monkeypatch):
        save_profile("old-user", _generate_wav_bytes(), "test")

        pdir = os.path.join(temp_profiles_dir, "old-user")
        meta_path = os.path.join(pdir, _METADATA_FILENAME)
        with open(meta_path) as f:
            meta = json.load(f)

        import datetime
        old_ts = (datetime.datetime.utcnow() - datetime.timedelta(hours=200)).isoformat()
        meta["updated_at"] = old_ts
        meta["created_at"] = old_ts
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        removed = cleanup_profiles(max_age_hours=168)
        assert removed == 1
        assert get_reference_path("old-user") is None
