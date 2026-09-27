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
        maximum_kbps: int = 96,
    ) -> dict[str, str | int | float]:
        if maximum_kbps < 1:
            raise ValueError("invalid_mp3_bitrate")
        bitrate = await self._native.run(
            self._delivery_bitrate_kbps,
            source,
            maximum_kbps,
            self._decode_timeout_seconds,
        )
        return await self._native.run(
            self._encode_mp3,
            source,
            target,
            bitrate,
            self._decode_timeout_seconds,
        )

    @staticmethod
    def _probe(path: Path, timeout_seconds: float) -> dict:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration,size,bit_rate,format_name",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=timeout_seconds,
        )
        return (json.loads(completed.stdout).get("format") or {})

    @classmethod
    def _delivery_bitrate_kbps(
        cls,
        source: Path,
        maximum_kbps: int,
        timeout_seconds: float,
    ) -> int:
        info = cls._probe(source, timeout_seconds)
        source_kbps = int(info.get("bit_rate") or 0) / 1000
        formats = set(str(info.get("format_name") or "").split(","))
        if source_kbps <= 0 or formats.intersection({"wav", "aiff", "flac"}):
            return maximum_kbps
        target = min(maximum_kbps, int(source_kbps * 0.8))
        return next(
            (rate for rate in (96, 80, 64, 56, 48, 40, 32, 24) if rate <= target),
            min(maximum_kbps, 24),
        )

    def _encode_mp3(
        self,
        source: Path,
        target: Path,
        bitrate_kbps: int,
        timeout_seconds: float,
    ) -> dict[str, str | int | float]:
        try:
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
                    "-b:a",
                    f"{bitrate_kbps}k",
                    str(target),
                ],
                capture_output=True,
                check=True,
                timeout=timeout_seconds,
            )
            source_info = AudioIO._probe(source, timeout_seconds)
            output_info = AudioIO._probe(target, timeout_seconds)
            source_duration = float(source_info.get("duration") or 0.0)
            output_duration = float(output_info.get("duration") or 0.0)
            tolerance = max(0.1, source_duration * 0.001)
            if abs(output_duration - source_duration) > tolerance:
                raise RuntimeError("encoded_audio_duration_mismatch")
            digest = hashlib.sha256()
            with target.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            return {
                "duration_seconds": output_duration,
                "size_bytes": int(output_info.get("size") or target.stat().st_size),
                "bitrate_bps": int(output_info.get("bit_rate") or 0),
                "bitrate_kbps": bitrate_kbps,
                "format": str(output_info.get("format_name") or ""),
                "sha256": digest.hexdigest(),
            }
        except BaseException:
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
