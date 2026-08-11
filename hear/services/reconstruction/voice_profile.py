"""User voice profile store for TTS voice cloning (Qwen3-TTS / Higgs).

Manages per-user reference audio clips stored on disk under the hear-ai tmp
directory.  Each profile has:
  - reference.wav   : the speaker's reference audio
  - reference.txt   : transcript of the reference audio
  - metadata.json   : creation time, duration, language, sample count

Profiles are cleaned up automatically by the existing temp sweep loop because
they live under ``hear_temp_directory()``.
"""

import json
import os
import time
import uuid
import wave
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

import torchaudio

from hear.config import settings
from hear.core.hear_temp import (
    hear_temp_directory,
)

_METADATA_FILENAME = "metadata.json"
_REFERENCE_AUDIO_FILENAME = "reference.wav"
_REFERENCE_TEXT_FILENAME = "reference.txt"


def _profiles_root() -> str:
    custom = (settings.VOICE_PROFILES_DIR or "").strip()
    if custom:
        path = os.path.abspath(os.path.expanduser(custom))
    else:
        path = os.path.join(hear_temp_directory(), "voice_profiles")
    os.makedirs(path, exist_ok=True)
    return path


def _profile_dir(user_id: str) -> str:
    safe_id = user_id.replace(os.sep, "_").replace("..", "_")
    path = os.path.join(_profiles_root(), safe_id)
    os.makedirs(path, exist_ok=True)
    return path


def _read_metadata(profile_path: str) -> dict:
    meta_path = os.path.join(profile_path, _METADATA_FILENAME)
    if not os.path.isfile(meta_path):
        return {}
    try:
        with open(meta_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _write_metadata(profile_path: str, meta: dict) -> None:
    meta_path = os.path.join(profile_path, _METADATA_FILENAME)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def save_profile(
    user_id: str,
    audio_bytes: bytes,
    transcript: str,
    *,
    language: Optional[str] = None,
    duration: Optional[float] = None,
) -> str:
    """Save or update a voice profile for *user_id*.

    Writes ``reference.wav``, ``reference.txt``, and ``metadata.json`` into
    ``{tmp}/voice_profiles/{user_id}/`` and registers the files with the temp
    tracking system.

    Parameters
    ----------
    user_id : str
        Unique identifier for the speaker / user.
    audio_bytes : bytes
        Raw audio bytes (WAV preferred; other formats accepted by torchaudio).
    transcript : str
        Transcript of the reference audio.
    language : str, optional
        Language code (e.g. ``"en"``).
    duration : float, optional
        Duration in seconds.  If not provided it is read from the WAV header.

    Returns
    -------
    str
        The profile directory path.
    """
    pdir = _profile_dir(user_id)

    ref_wav = os.path.join(pdir, _REFERENCE_AUDIO_FILENAME)
    with open(ref_wav, "wb") as f:
        f.write(audio_bytes)


    ref_txt = os.path.join(pdir, _REFERENCE_TEXT_FILENAME)
    with open(ref_txt, "w") as f:
        f.write(transcript.strip())


    actual_duration = duration
    if actual_duration is None:
        try:
            info = torchaudio.info(ref_wav)
            actual_duration = round(info.num_frames / info.sample_rate, 3)
        except Exception:
            try:
                with wave.open(ref_wav, "rb") as wav_file:
                    actual_duration = round(
                        wav_file.getnframes() / wav_file.getframerate(), 3
                    )
            except (OSError, EOFError, wave.Error, ZeroDivisionError):
                actual_duration = 0.0

    meta = _read_metadata(pdir)
    meta.update({
        "user_id": user_id,
        "created_at": meta.get("created_at", datetime.now(UTC).isoformat()),
        "updated_at": datetime.now(UTC).isoformat(),
        "duration": actual_duration,
        "language": language or meta.get("language", "en"),
        "sample_count": meta.get("sample_count", 0) + 1,
    })
    _write_metadata(pdir, meta)

    return pdir


def get_reference_path(user_id: str) -> Optional[str]:
    """Return the reference ``.wav`` path for *user_id*, or ``None``.

    The caller is responsible for checking that the file still exists before
    passing it to Higgs Audio.
    """
    pdir = _profile_dir(user_id)
    ref_wav = os.path.join(pdir, _REFERENCE_AUDIO_FILENAME)
    ref_txt = os.path.join(pdir, _REFERENCE_TEXT_FILENAME)
    if os.path.isfile(ref_wav) and os.path.isfile(ref_txt):
        return ref_wav
    return None


def get_reference_text(user_id: str) -> Optional[str]:
    """Return the transcript for the stored voice profile, or ``None``."""
    pdir = _profile_dir(user_id)
    ref_txt = os.path.join(pdir, _REFERENCE_TEXT_FILENAME)
    if not os.path.isfile(ref_txt):
        return None
    try:
        with open(ref_txt, "r") as f:
            return f.read().strip()
    except OSError:
        return None


def get_or_create_from_track(
    user_id: str,
    audio_path: str,
    *,
    transcript: Optional[str] = None,
) -> str:
    """Create a voice profile from a track audio file if one does not exist.

    If the user already has a profile, returns the existing directory.
    Otherwise reads the audio file, builds the profile, and returns the path.

    Parameters
    ----------
    user_id : str
        Unique identifier for the speaker / user.
    audio_path : str
        Path to the source audio file (WAV).
    transcript : str, optional
        Transcript text.  If not provided, an empty string is used (Higgs will
        still clone voice but with less precision).

    Returns
    -------
    str
        The profile directory path.
    """
    existing = get_reference_path(user_id)
    if existing:
        return _profile_dir(user_id)

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    return save_profile(
        user_id,
        audio_bytes,
        transcript=transcript or "",
    )


def list_profiles() -> list[dict]:
    """Return metadata for all stored voice profiles."""
    root = _profiles_root()
    profiles: list[dict] = []
    if not os.path.isdir(root):
        return profiles

    for name in os.listdir(root):
        pdir = os.path.join(root, name)
        if not os.path.isdir(pdir):
            continue
        meta = _read_metadata(pdir)
        if meta:
            profiles.append(meta)

    return sorted(profiles, key=lambda m: m.get("updated_at", ""), reverse=True)


def cleanup_profiles(max_age_hours: Optional[int] = None) -> int:
    """Remove voice profiles older than *max_age_hours*.

    Defaults to ``settings.VOICE_PROFILE_MAX_AGE_HOURS``.

    Returns
    -------
    int
        Number of profiles removed.
    """
    max_hours = max_age_hours if max_age_hours is not None else settings.VOICE_PROFILE_MAX_AGE_HOURS
    cutoff = time.time() - (max_hours * 3600)
    root = _profiles_root()
    removed = 0

    if not os.path.isdir(root):
        return removed

    for name in os.listdir(root):
        pdir = os.path.join(root, name)
        if not os.path.isdir(pdir):
            continue
        meta = _read_metadata(pdir)
        updated = meta.get("updated_at", meta.get("created_at", ""))
        if not updated:
            try:
                updated = datetime.fromtimestamp(
                    os.path.getmtime(os.path.join(pdir, _REFERENCE_AUDIO_FILENAME))
                ).isoformat()
            except OSError:
                continue

        try:
            updated_ts = datetime.fromisoformat(updated).timestamp()
        except (ValueError, TypeError):
            continue

        if updated_ts < cutoff:
            try:
                for fname in os.listdir(pdir):
                    os.remove(os.path.join(pdir, fname))
                os.rmdir(pdir)
                removed += 1
            except OSError:
                pass

    return removed
