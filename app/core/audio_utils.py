import os
import subprocess
import tempfile

import torch
import torchaudio

from app.core.hear_temp import hear_temp_directory


def save_as_mp3(waveform: torch.Tensor, sample_rate: int, bitrate_kbps: int = 192) -> str:
    tmp_root = hear_temp_directory()
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

    return mp3_path


def convert_wav_file_to_mp3(wav_path: str, bitrate_kbps: int = 192) -> str:
    tmp_root = hear_temp_directory()
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
    return mp3_path
