from datetime import UTC, datetime, timedelta
from pathlib import Path

from hear.contracts.jobs import ArtifactStorage
from hear.storage.b2 import B2Storage


class FakeS3:
    def __init__(self) -> None:
        self.uploaded = None
        self.head = {
            "ContentLength": 4,
            "Metadata": {
                "sha256": "a" * 64,
            },
        }
        self.read_back_called = False

    def upload_file(self, filename, bucket, key, ExtraArgs=None, Config=None):
        self.uploaded = (filename, bucket, key, ExtraArgs, Config)

    def head_object(self, Bucket, Key):
        return self.head

    def get_object(self, Bucket, Key):
        self.read_back_called = True
        raise AssertionError("read back must not happen")


class TestB2Storage:
    def test_upload_verifies_head_without_reading_object(self, tmp_path: Path, monkeypatch):
        fake = FakeS3()
        monkeypatch.setattr("hear.storage.b2.boto3.client", lambda *args, **kwargs: fake)
        context = ArtifactStorage(
            endpoint_url="https://s3.example.com",
            bucket_name="bucket",
            key_id="key",
            application_key="secret",
            folder_prefix="users/user-1/",
            public_base_url="https://cdn.example.com/media",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )
        local = tmp_path / "a.mp3"
        local.write_bytes(b"test")
        storage = B2Storage(context)
        artifact = storage.upload_file(
            local,
            storage.key("jobs", "a.mp3"),
            sha256="a" * 64,
            content_type="audio/mpeg",
        )
        assert artifact.size_bytes == 4
        assert fake.read_back_called is False