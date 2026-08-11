import asyncio
import os
import subprocess
from typing import Optional

import httpx

from hear.core.hear_temp import hear_temp_job_dir, hear_temp_standalone_dir


async def download_audio(
    url: str,
    suffix: str = ".wav",
    *,
    db=None,
    job_id: Optional[str] = None,
    run_id: Optional[str] = None,
    track_id: Optional[str] = None,
    purpose: str = "audio",
    convert_to_wav: bool = False,
) -> str:
    if job_id and run_id:
        dir_path = hear_temp_job_dir(job_id, run_id)
    else:
        dir_path = hear_temp_standalone_dir(purpose)
    path = os.path.join(dir_path, f"{purpose}{suffix}")
    download_path = f"{path}.source" if convert_to_wav else path
    partial_path = f"{download_path}.part"
    downloaded = 0
    timeout = httpx.Timeout(connect=15, read=None, write=30, pool=30)
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=timeout,
        ) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                headers = getattr(response, "headers", None)
                expected_length = headers.get("content-length") if headers else None
                with open(partial_path, "wb") as output:
                    async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        output.write(chunk)
                        downloaded += len(chunk)
        if downloaded == 0:
            raise ValueError("audio download returned an empty response")
        if expected_length is not None and downloaded != int(expected_length):
            raise ValueError(
                "audio download was truncated: "
                f"expected {expected_length} bytes, received {downloaded}"
            )
        os.replace(partial_path, download_path)
        if convert_to_wav:
            await asyncio.to_thread(_convert_to_wav, download_path, path)
            os.unlink(download_path)
        return path
    except Exception:
        for candidate in (partial_path, download_path, path):
            try:
                os.unlink(candidate)
            except OSError:
                pass
        raise


def _convert_to_wav(source_path: str, wav_path: str) -> None:
    """Decode a downloaded audio file into a real mono PCM WAV file."""
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-y", "-i", source_path,
            "-vn", "-ac", "1", "-c:a", "pcm_s16le", wav_path,
        ],
        capture_output=True,
        check=True,
    )
