import asyncio
import json
import os
import subprocess
from typing import Optional

import soundfile as sf
import numpy as np
import torch

from hear.core.hear_temp import hear_temp_job_dir, hear_temp_standalone_dir


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
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, _convert_sync, wav_path, mp3_path, bitrate_kbps,
    )
    source = probe_audio(wav_path)
    output = probe_audio(mp3_path)
    duration_delta = abs(output["duration_seconds"] - source["duration_seconds"])
    tolerance = max(1.0, source["duration_seconds"] * 0.001)
    if duration_delta > tolerance:
        try:
            os.unlink(mp3_path)
        except OSError:
            pass
        raise RuntimeError(
            "encoded audio duration mismatch: "
            f"source={source['duration_seconds']:.3f}s, "
            f"output={output['duration_seconds']:.3f}s"
        )
    return mp3_path


def _convert_sync(wav_path: str, mp3_path: str, bitrate_kbps: int) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", wav_path, "-b:a", f"{bitrate_kbps}k", mp3_path],
        capture_output=True, check=True,
    )


def probe_audio(path: str) -> dict[str, float | int | str]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size,bit_rate,format_name",
            "-of",
            "json",
            path,
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    audio_format = payload.get("format") or {}
    return {
        "duration_seconds": float(audio_format.get("duration") or 0.0),
        "size_bytes": int(audio_format.get("size") or os.path.getsize(path)),
        "bitrate_bps": int(audio_format.get("bit_rate") or 0),
        "format": str(audio_format.get("format_name") or ""),
    }


def delivery_bitrate_kbps(
    source_path: str,
    *,
    maximum_kbps: int = 96,
    reduction_ratio: float = 0.8,
) -> int:
    """Choose an Alexa delivery bitrate without upscaling compressed input."""
    source = probe_audio(source_path)
    source_kbps = int(source["bitrate_bps"]) / 1000
    formats = set(str(source["format"]).split(","))
    if source_kbps <= 0 or formats.intersection({"wav", "aiff", "flac"}):
        return maximum_kbps

    target = min(maximum_kbps, int(source_kbps * reduction_ratio))
    ladder = (96, 80, 64, 56, 48, 40, 32, 24)
    return next((rate for rate in ladder if rate <= target), 24)


def save_as_mp3(
    audio,
    sample_rate: int,
    *,
    job_id: Optional[str] = None,
    run_id: Optional[str] = None,
    track_id: Optional[str] = None,
    purpose: str = "mp3",
    bitrate_kbps: int = 96,
) -> str:
    if isinstance(audio, torch.Tensor):
        audio_np = audio.detach().cpu().numpy()
    else:
        audio_np = np.asarray(audio)

    if audio_np.ndim == 1:
        audio_np = audio_np.reshape(1, -1)
    elif audio_np.ndim == 2 and audio_np.shape[0] > audio_np.shape[1]:
        audio_np = audio_np.T

    if job_id and run_id:
        output_dir = hear_temp_job_dir(job_id, run_id)
    else:
        output_dir = hear_temp_standalone_dir(purpose)
    wav_path = os.path.join(output_dir, f"{purpose}.wav")

    sf.write(wav_path, audio_np.T, sample_rate, format="WAV")

    mp3_path = wav_path + ".mp3"
    _convert_sync(wav_path, mp3_path, bitrate_kbps)

    try:
        os.unlink(wav_path)
    except OSError:
        pass

    return mp3_path
