import os
import subprocess
import tempfile

import torch
import torchaudio


def save_as_mp3(waveform: torch.Tensor, sample_rate: int, bitrate_kbps: int = 192) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        wav_path = tmp_wav.name

    mp3_fd, mp3_path = tempfile.mkstemp(suffix=".mp3")
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
