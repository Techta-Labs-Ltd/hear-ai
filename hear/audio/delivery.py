from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from hear.execution.native import NativeExecutor


class AudioDelivery:
    def __init__(
        self,
        native: NativeExecutor,
        *,
        encode_timeout_seconds: float = 1200.0,
    ) -> None:
        self._native = native
        self._encode_timeout_seconds = encode_timeout_seconds

    async def encode_mp3(
        self,
        source: Path,
        destination: Path,
        *,
        maximum_kbps: int,
    ) -> tuple[int, dict, dict, str]:
        source_info = await self._native.run(self._probe, source)
        bitrate = self._bitrate(source_info, maximum_kbps)
        await self._native.run(self._encode, source, destination, bitrate)
        output_info = await self._native.run(self._probe, destination)
        delta = abs(float(output_info["duration_seconds"]) - float(source_info["duration_seconds"]))
        tolerance = max(1.0, float(source_info["duration_seconds"]) * 0.001)
        if delta > tolerance:
            await self._native.run(destination.unlink, missing_ok=True)
            raise RuntimeError("encoded_audio_duration_mismatch")
        digest = await self._native.run(self._sha256, destination)
        return bitrate, source_info, output_info, digest

    def _encode(
        self,
        source: Path,
        destination: Path,
        bitrate_kbps: int,
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
                "-b:a",
                f"{bitrate_kbps}k",
                str(destination),
            ],
            capture_output=True,
            check=True,
            timeout=self._encode_timeout_seconds,
        )

    @staticmethod
    def _probe(path: Path) -> dict[str, float | int | str]:
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
        )
        payload = json.loads(completed.stdout)
        audio_format = payload.get("format") or {}
        return {
            "duration_seconds": float(audio_format.get("duration") or 0.0),
            "size_bytes": int(audio_format.get("size") or os.path.getsize(path)),
            "bitrate_bps": int(audio_format.get("bit_rate") or 0),
            "format": str(audio_format.get("format_name") or ""),
        }

    @staticmethod
    def _bitrate(source: dict, maximum_kbps: int) -> int:
        source_kbps = int(source["bitrate_bps"]) / 1000
        formats = set(str(source["format"]).split(","))
        if source_kbps <= 0 or formats.intersection({"wav", "aiff", "flac"}):
            return maximum_kbps
        target = min(maximum_kbps, int(source_kbps * 0.8))
        ladder = (96, 80, 64, 56, 48, 40, 32, 24)
        return next((rate for rate in ladder if rate <= target), 24)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
