import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hear.audio.io import AudioIO


class InlineNativeExecutor:
    async def run(self, function, *args, **kwargs):
        return function(*args, **kwargs)


def test_delivery_bitrate_respects_source_encoding_and_limit(monkeypatch):
    formats = {
        "source.wav": {
            "bit_rate": "1411200",
            "format_name": "wav",
        },
        "source.mp3": {
            "bit_rate": "160000",
            "format_name": "mp3",
        },
        "source.low.mp3": {
            "bit_rate": "64000",
            "format_name": "mp3",
        },
    }
    monkeypatch.setattr(
        AudioIO,
        "_probe",
        staticmethod(lambda path, timeout: formats[path.name]),
    )

    assert AudioIO._delivery_bitrate_kbps(Path("source.wav"), 80, 1) == 80
    assert AudioIO._delivery_bitrate_kbps(Path("source.wav"), 16, 1) == 16
    assert AudioIO._delivery_bitrate_kbps(Path("source.mp3"), 96, 1) == 96
    assert AudioIO._delivery_bitrate_kbps(Path("source.low.mp3"), 96, 1) == 48


@pytest.mark.anyio
async def test_encode_mp3_returns_verified_metadata(monkeypatch, tmp_path: Path):
    source = tmp_path / "source.wav"
    target = tmp_path / "delivery.mp3"
    source.write_bytes(b"source")
    formats = {
        source: {
            "duration": "12.5",
            "size": "6",
            "bit_rate": "1411200",
            "format_name": "wav",
        },
        target: {
            "duration": "12.51",
            "size": "7",
            "bit_rate": "96000",
            "format_name": "mp3",
        },
    }
    timeouts = []

    def fake_run(command, **kwargs):
        if command[0] == "ffmpeg":
            target.write_bytes(b"encoded")
            timeouts.append(kwargs["timeout"])
            return SimpleNamespace(stdout="")
        payload = {"format": formats[Path(command[-1])]}
        return SimpleNamespace(stdout=json.dumps(payload))

    monkeypatch.setattr("hear.audio.io.subprocess.run", fake_run)
    audio = AudioIO(None, InlineNativeExecutor(), max_download_bytes=100, decode_timeout_seconds=9)

    result = await audio.encode_mp3(source, target, maximum_kbps=96)

    assert result == {
        "duration_seconds": 12.51,
        "size_bytes": 7,
        "bitrate_bps": 96000,
        "bitrate_kbps": 96,
        "format": "mp3",
        "sha256": hashlib.sha256(b"encoded").hexdigest(),
    }
    assert timeouts == [9]


def test_encode_mp3_removes_output_when_duration_validation_fails(monkeypatch, tmp_path: Path):
    source = tmp_path / "source.wav"
    target = tmp_path / "delivery.mp3"
    source.write_bytes(b"source")
    monkeypatch.setattr(
        "hear.audio.io.subprocess.run",
        lambda command, **kwargs: (target.write_bytes(b"encoded"), SimpleNamespace())[1],
    )
    durations = iter(({"duration": "10"}, {"duration": "20"}))
    monkeypatch.setattr(AudioIO, "_probe", staticmethod(lambda path, timeout: next(durations)))
    audio = AudioIO(None, InlineNativeExecutor(), max_download_bytes=100, decode_timeout_seconds=5)

    with pytest.raises(RuntimeError, match="encoded_audio_duration_mismatch"):
        audio._encode_mp3(source, target, 96, 5)

    assert not target.exists()
