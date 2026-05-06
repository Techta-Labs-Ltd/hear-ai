import os
import subprocess
import tempfile
from typing import Optional

import torch
import torchaudio

from app.core.hear_temp import (
    hear_temp_directory,
    hear_temp_job_dir,
    register_temp_standalone,
)


def _temp_dir_for(job_id: Optional[str], run_id: Optional[str]) -> str:
    if job_id and run_id:
        return hear_temp_job_dir(job_id, run_id)
    return hear_temp_directory()


def save_as_mp3(
    waveform: torch.Tensor,
    sample_rate: int,
    bitrate_kbps: int = 192,
    *,
    job_id: Optional[str] = None,
    run_id: Optional[str] = None,
    track_id: Optional[str] = None,
    purpose: str = "mp3_export",
) -> str:
    tmp_root = _temp_dir_for(job_id, run_id)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=tmp_root) as tmp_wav:
        wav_path = tmp_wav.name

    mp3_fd, mp3_path = tempfile.mkstemp(suffix=".mp3", dir=tmp_root)
    os.close(mp3_fd)

    try:
        torchaudio.save(wav_path, waveform, sample_rate)
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", wav_path,
                "-b:a", f"{bitrate_kbps}k",
                "-q:a", "2",
                mp3_path,
            ],
            check=True,
            capture_output=True,
        )
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)

    register_temp_standalone(
        mp3_path,
        purpose=purpose,
        job_id=job_id,
        run_id=run_id,
        track_id=track_id,
    )
    return mp3_path


def convert_wav_file_to_mp3(
    wav_path: str,
    bitrate_kbps: int = 192,
    *,
    job_id: Optional[str] = None,
    run_id: Optional[str] = None,
    track_id: Optional[str] = None,
    purpose: str = "mp3_compress",
) -> str:
    tmp_root = _temp_dir_for(job_id, run_id)
    mp3_fd, mp3_path = tempfile.mkstemp(suffix=".mp3", dir=tmp_root)
    os.close(mp3_fd)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            wav_path,
            "-c:a",
            "libmp3lame",
            "-b:a",
            f"{bitrate_kbps}k",
            mp3_path,
        ],
        check=True,
        capture_output=True,
    )
    register_temp_standalone(
        mp3_path,
        purpose=purpose,
        job_id=job_id,
        run_id=run_id,
        track_id=track_id,
    )
    return mp3_path
