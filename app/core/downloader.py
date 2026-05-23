import os
import subprocess
import tempfile
from typing import Optional
from urllib.parse import urlparse

import httpx
import torchaudio

from app.core.hear_temp import (
    drop_temp_standalone,
    hear_temp_directory,
    hear_temp_job_dir,
    register_temp,
    register_temp_standalone,
)

AUDIO_CONTENT_TYPES = {
    "audio/wav", "audio/wave", "audio/x-wav",
    "audio/mpeg", "audio/mp3", "audio/mp4",
    "audio/ogg", "audio/flac", "audio/aac",
    "audio/webm", "application/octet-stream", "binary/octet-stream",
}

_CONTENT_TYPE_TO_EXT = {
    "audio/wav":    ".wav",
    "audio/wave":   ".wav",
    "audio/x-wav":  ".wav",
    "audio/mpeg":   ".mp3",
    "audio/mp3":    ".mp3",
    "audio/mp4":    ".m4a",
    "audio/ogg":    ".ogg",
    "audio/flac":   ".flac",
    "audio/aac":    ".aac",
    "audio/webm":   ".webm",
}


def _ensure_https(url: str) -> str:
    if not url or not isinstance(url, str):
        raise ValueError("audio_url is missing or invalid")
    if url.startswith("http://") and "127.0.0.1" not in url and "localhost" not in url:
        return "https://" + url[7:]
    return url


def _detect_suffix(url: str, content_type: str) -> str:
    path = urlparse(url).path
    _, ext = os.path.splitext(path)
    if ext.lower() in {".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a", ".webm", ".mp4"}:
        return ext.lower()
    return _CONTENT_TYPE_TO_EXT.get(content_type, ".wav")


def _temp_dir_for(job_id: Optional[str], run_id: Optional[str]) -> str:
    if job_id and run_id:
        return hear_temp_job_dir(job_id, run_id)
    return hear_temp_directory()


def _register(
    db,
    path: str,
    *,
    purpose: str,
    job_id: Optional[str],
    run_id: Optional[str],
    track_id: Optional[str],
) -> None:
    if db is not None:
        register_temp(
            db, path, purpose=purpose, job_id=job_id, run_id=run_id, track_id=track_id
        )
    else:
        register_temp_standalone(
            path, purpose=purpose, job_id=job_id, run_id=run_id, track_id=track_id
        )


async def download_audio(
    url: str,
    suffix: Optional[str] = None,
    *,
    db=None,
    job_id: Optional[str] = None,
    run_id: Optional[str] = None,
    track_id: Optional[str] = None,
    purpose: str = "download",
) -> str:
    url = _ensure_https(url)
    tmp_root = _temp_dir_for(job_id, run_id)

    async with httpx.AsyncClient(timeout=300, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").split(";")[0].strip()
        if content_type and content_type not in AUDIO_CONTENT_TYPES and "audio" not in content_type:
            raise ValueError(
                f"Expected audio file but got content-type '{content_type}' from {url}"
            )

        body = response.content
        if len(body) == 0:
            raise ValueError(f"Downloaded file is empty from {url}")

        expected_len = response.headers.get("content-length")
        if expected_len and expected_len.isdigit() and int(expected_len) != len(body):
            raise ValueError(
                f"Truncated download from {url}: got {len(body)} bytes, expected {expected_len}"
            )

        resolved_suffix = suffix or _detect_suffix(url, content_type)

        tmp = tempfile.NamedTemporaryFile(suffix=resolved_suffix, delete=False, dir=tmp_root)
        tmp.write(body)
        tmp.flush()
        tmp.close()

        if os.path.getsize(tmp.name) != len(body):
            os.unlink(tmp.name)
            raise ValueError(f"Written temp file size mismatch for {url}")

        target_suffix = (suffix or ".wav").lower()
        if target_suffix != ".wav":
            _register(
                db,
                tmp.name,
                purpose=purpose,
                job_id=job_id,
                run_id=run_id,
                track_id=track_id,
            )
            return tmp.name

        wav_path = _convert_to_wav(
            tmp.name,
            db=db,
            job_id=job_id,
            run_id=run_id,
            track_id=track_id,
        )
        if wav_path != tmp.name and os.path.exists(tmp.name):
            os.unlink(tmp.name)
        _register(
            db,
            wav_path,
            purpose=f"{purpose}_wav",
            job_id=job_id,
            run_id=run_id,
            track_id=track_id,
        )
        return wav_path


def cleanup_temp(path: str):
    drop_temp_standalone(path)


def _convert_to_wav(
    path: str,
    *,
    db=None,
    job_id: Optional[str] = None,
    run_id: Optional[str] = None,
    track_id: Optional[str] = None,
) -> str:
    tmp_root = _temp_dir_for(job_id, run_id)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=tmp_root) as tmp_wav:
        wav_path = tmp_wav.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-vn", "-ac", "1", "-ar", "44100", wav_path],
            check=True,
            capture_output=True,
        )
    except Exception:
        if os.path.exists(wav_path):
            os.unlink(wav_path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=tmp_root) as fallback_wav:
            wav_path = fallback_wav.name
        try:
            waveform, sr = torchaudio.load(path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != 44100:
                waveform = torchaudio.functional.resample(waveform, sr, 44100)
            torchaudio.save(wav_path, waveform, 44100)
        except Exception as exc:
            if os.path.exists(wav_path):
                os.unlink(wav_path)
            raise ValueError(f"Failed to normalize audio to wav: {exc}") from exc
    if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
        if os.path.exists(wav_path):
            os.unlink(wav_path)
        raise ValueError("Failed to normalize audio to wav: empty output")
    return wav_path
