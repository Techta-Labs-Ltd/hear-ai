from __future__ import annotations

import os
import subprocess
from pathlib import Path

import httpx

from hear.audio.workspace import AudioWorkspace
from hear.execution.native import NativeExecutor


class AudioIO:
    def __init__(
        self,
        client: httpx.AsyncClient,
        native: NativeExecutor,
        *,
        max_download_bytes: int,
        decode_timeout_seconds: float,
    ) -> None:
        self._client = client
        self._native = native
        self._max_download_bytes = max_download_bytes
        self._decode_timeout_seconds = decode_timeout_seconds

    async def download_to_wav(
        self,
        url: str,
        workspace: AudioWorkspace,
        *,
        preserve_channels: bool = True,
    ) -> Path:
        source = workspace.file("source.audio")
        partial = workspace.file("source.audio.part")
        target = workspace.file("source.wav")
        downloaded = 0
        expected_length: int | None = None
        try:
            async with self._client.stream("GET", url) as response:
                response.raise_for_status()
                raw_length = response.headers.get("content-length")
                if raw_length is not None:
                    expected_length = int(raw_length)
                    if expected_length > self._max_download_bytes:
                        raise ValueError("source_audio_too_large")
                with partial.open("wb") as output:
                    async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        downloaded += len(chunk)
                        if downloaded > self._max_download_bytes:
                            raise ValueError("source_audio_too_large")
                        output.write(chunk)
            if downloaded == 0:
                raise ValueError("empty_source_audio")
            if expected_length is not None and downloaded != expected_length:
                raise ValueError("truncated_source_audio")
            os.replace(partial, source)
            await self._native.run(
                self._convert_to_wav,
                source,
                target,
                preserve_channels,
            )
            source.unlink(missing_ok=True)
            return target
        except BaseException:
            partial.unlink(missing_ok=True)
            source.unlink(missing_ok=True)
            target.unlink(missing_ok=True)
            raise

    def _convert_to_wav(
        self,
        source: Path,
        target: Path,
        preserve_channels: bool,
    ) -> None:
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                str(source),
                "-vn",
                *([] if preserve_channels else ["-ac", "1"]),
                "-c:a",
                "pcm_s16le",
                "-rf64",
                "auto",
                str(target),
            ],
            capture_output=True,
            check=True,
            timeout=self._decode_timeout_seconds,
        )