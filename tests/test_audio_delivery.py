import asyncio
from datetime import datetime, timedelta, timezone

import numpy as np
import soundfile as sf

from hear.core import storage
from hear.models.schemas import StorageContext
from hear.core.audio_utils import (
    convert_wav_file_to_mp3,
    delivery_bitrate_kbps,
    probe_audio,
)


def storage_context():
    return StorageContext(
        endpoint_url="https://s3.example.test",
        bucket_name="bucket-a",
        key_id="key-id",
        application_key="application-key",
        folder_prefix="users/user/jobs/job",
        public_base_url="https://cdn.example.test",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )


class RecordingS3Client:
    def __init__(self, remote_size: int):
        self.remote_size = remote_size
        self.upload = None

    def upload_file(self, local_path, bucket, key, ExtraArgs):
        self.upload = (local_path, bucket, key, ExtraArgs)

    def head_object(self, Bucket, Key):
        return {"ContentLength": self.remote_size}


def test_mp3_upload_uses_mpeg_content_type_and_verifies_size(tmp_path, monkeypatch):
    audio = tmp_path / "clean.mp3"
    audio.write_bytes(b"mp3-data")
    client = RecordingS3Client(audio.stat().st_size)
    monkeypatch.setattr(storage.boto3, "client", lambda *args, **kwargs: client)

    uploaded = storage.B2Storage(storage_context())
    uploaded.upload_file(str(audio), uploaded.key("enhanced", "job.mp3"))

    assert client.upload[3] == {"ContentType": "audio/mpeg"}


def test_upload_rejects_remote_size_mismatch(tmp_path, monkeypatch):
    audio = tmp_path / "clean.mp3"
    audio.write_bytes(b"mp3-data")
    client = RecordingS3Client(audio.stat().st_size - 1)
    monkeypatch.setattr(storage.boto3, "client", lambda *args, **kwargs: client)
    uploaded = storage.B2Storage(storage_context())

    try:
        uploaded.upload_file(str(audio), uploaded.key("enhanced", "job.mp3"))
    except RuntimeError as exc:
        assert "size mismatch" in str(exc)
    else:
        raise AssertionError("size mismatch must fail the upload")


def test_delivery_bitrate_reduces_compressed_source(monkeypatch):
    monkeypatch.setattr(
        "hear.core.audio_utils.probe_audio",
        lambda _path: {"bitrate_bps": 80_000, "format": "mp3"},
    )

    assert delivery_bitrate_kbps("source.mp3") == 64


def test_delivery_bitrate_caps_lossless_source(monkeypatch):
    monkeypatch.setattr(
        "hear.core.audio_utils.probe_audio",
        lambda _path: {"bitrate_bps": 1_536_000, "format": "wav"},
    )

    assert delivery_bitrate_kbps("source.wav") == 96


def test_mp3_conversion_preserves_duration_and_reports_size(tmp_path):
    source = tmp_path / "source.wav"
    samples = np.zeros(48_000 * 2, dtype=np.float32)
    sf.write(source, samples, 48_000)

    output = asyncio.run(convert_wav_file_to_mp3(str(source), bitrate_kbps=96))
    source_info = probe_audio(str(source))
    output_info = probe_audio(output)

    assert abs(output_info["duration_seconds"] - source_info["duration_seconds"]) <= 1
    assert output_info["size_bytes"] > 0
    assert "mp3" in output_info["format"]
