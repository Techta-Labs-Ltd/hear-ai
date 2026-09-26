from __future__ import annotations

import hashlib
import json
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

    async def download_source(
        self,
        url: str,
        workspace: AudioWorkspace,
        *,
        name: str = "source.audio",
    ) -> Path:
        source = workspace.file(name)
        partial = workspace.file(f"{name}.part")
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
            return source
        except BaseException:
            partial.unlink(missing_ok=True)
            source.unlink(missing_ok=True)
            raise

    async def convert_to_wav(
        self,
        source: Path,
        workspace: AudioWorkspace,
        *,
        preserve_channels: bool = True,
        name: str = "source.wav",
    ) -> Path:
        target = workspace.file(name)
        try:
            await self._native.run(
                self._convert_to_wav,
                source,
                target,
                preserve_channels,
            )
            return target
        except BaseException:
            target.unlink(missing_ok=True)
            raise

    async def download_to_wav(
        self,
        url: str,
        workspace: AudioWorkspace,
        *,
        preserve_channels: bool = True,
    ) -> Path:
        source = await self.download_source(url, workspace)
        target = await self.convert_to_wav(
            source,
            workspace,
            preserve_channels=preserve_channels,
        )
        source.unlink(missing_ok=True)
        return target


    async def encode_mp3(
        self,
        source: Path,
        target: Path,
        *,
        bitrate_kbps: int,
    ) -> dict:
        return await self._native.run(
            self._encode_mp3,
            source,
            target,
            bitrate_kbps,
        )

    @staticmethod
    def _encode_mp3(
        source: Path,
        target: Path,
        bitrate_kbps: int,
    ) -> dict:
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                str(source),
                "-b:a",
                f"{bitrate_kbps}k",
                str(target),
            ],
            capture_output=True,
            check=True,
        )
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration,size,bit_rate,format_name",
                "-of",
                "json",
                str(target),
            ],
            capture_output=True,
            check=True,
            text=True,
        )
        payload = json.loads(completed.stdout).get("format") or {}
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        return {
            "duration_seconds": float(payload.get("duration") or 0.0),
            "size_bytes": int(payload.get("size") or target.stat().st_size),
            "bitrate_bps": int(payload.get("bit_rate") or 0),
            "format": str(payload.get("format_name") or ""),
            "sha256": digest.hexdigest(),
        }

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