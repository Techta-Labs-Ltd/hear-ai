import os
import subprocess

import httpx

from hear.config import settings
from hear.core.blocking import AsyncCompletion
from hear.core.hear_temp import TempWorkspace


class AudioDownloader:
    @staticmethod
    async def download_audio(
        url: str,
        suffix: str = ".wav",
        *,
        db=None,
        job_id: str | None = None,
        run_id: str | None = None,
        track_id: str | None = None,
        purpose: str = "audio",
        convert_to_wav: bool = False,
        preserve_channels: bool = False,
    ) -> str:
        if job_id and run_id:
            dir_path = TempWorkspace.hear_temp_job_dir(job_id, run_id)
        else:
            dir_path = TempWorkspace.hear_temp_standalone_dir(purpose)
        path = os.path.join(dir_path, f"{purpose}{suffix}")
        download_path = f"{path}.source" if convert_to_wav else path
        partial_path = f"{download_path}.part"
        downloaded = 0
        timeout = httpx.Timeout(
            connect=15, read=settings.AUDIO_DOWNLOAD_READ_TIMEOUT_SECONDS, write=30, pool=30
        )
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    headers = getattr(response, "headers", None)
                    expected_length = headers.get("content-length") if headers else None
                    if (
                        expected_length is not None
                        and int(expected_length) > settings.AUDIO_DOWNLOAD_MAX_BYTES
                    ):
                        raise ValueError("source_audio_too_large")
                    with open(partial_path, "wb") as output:
                        async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                            if not chunk:
                                continue
                            if downloaded + len(chunk) > settings.AUDIO_DOWNLOAD_MAX_BYTES:
                                raise ValueError("source_audio_too_large")
                            output.write(chunk)
                            downloaded += len(chunk)
            if downloaded == 0:
                raise ValueError("audio download returned an empty response")
            if expected_length is not None and downloaded != int(expected_length):
                raise ValueError(
                    f"audio download was truncated: expected {expected_length} bytes, received {downloaded}"
                )
            os.replace(partial_path, download_path)
            if convert_to_wav:
                await AsyncCompletion.run_blocking_to_completion(
                    lambda: AudioDownloader._convert_to_wav(
                        download_path, path, preserve_channels=preserve_channels
                    )
                )
                os.unlink(download_path)
            return path
        except BaseException:
            for candidate in (partial_path, download_path, path):
                try:
                    os.unlink(candidate)
                except OSError:
                    pass
            raise

    @staticmethod
    def _convert_to_wav(
        source_path: str, wav_path: str, *, preserve_channels: bool = False
    ) -> None:
        """Decode a downloaded audio file into a real mono PCM WAV file."""
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                source_path,
                "-vn",
                *([] if preserve_channels else ["-ac", "1"]),
                "-c:a",
                "pcm_s16le",
                "-rf64",
                "auto",
                wav_path,
            ],
            capture_output=True,
            check=True,
            timeout=settings.AUDIO_DECODE_TIMEOUT_SECONDS,
        )
