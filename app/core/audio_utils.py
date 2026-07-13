import os
import subprocess
import tempfile
from typing import Optional

import soundfile as sf
import numpy as np


async def convert_wav_file_to_mp3(
    wav_path: str,
    bitrate_kbps: int = 96,
    *,
    job_id: Optional[str] = None,
    run_id: Optional[str] = None,
    track_id: Optional[str] = None,
    purpose: str = "mp3",
) -> str:
    mp3_path = wav_path + f"_{purpose}.mp3"
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, _convert_sync, wav_path, mp3_path, bitrate_kbps,
    )
    return mp3_path


def _convert_sync(wav_path: str, mp3_path: str, bitrate_kbps: int) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", wav_path, "-b:a", f"{bitrate_kbps}k", mp3_path],
        capture_output=True, check=True,
    )


def save_as_mp3(
    audio,
    sample_rate: int,
    *,
    job_id: Optional[str] = None,
    run_id: Optional[str] = None,
    track_id: Optional[str] = None,
    purpose: str = "mp3",
) -> str:
    import torch

    if isinstance(audio, torch.Tensor):
        audio_np = audio.detach().cpu().numpy()
    else:
        audio_np = np.asarray(audio)

    if audio_np.ndim == 1:
        audio_np = audio_np.reshape(1, -1)
    elif audio_np.ndim == 2 and audio_np.shape[0] > audio_np.shape[1]:
        audio_np = audio_np.T

    wav_path = os.path.join(tempfile.mkdtemp(prefix="hear_save_"), f"{purpose}.wav")

    sf.write(wav_path, audio_np.T, sample_rate, format="WAV")

    mp3_path = wav_path + ".mp3"
    _convert_sync(wav_path, mp3_path, 96)

    try:
        os.unlink(wav_path)
    except OSError:
        pass

    return mp3_path
