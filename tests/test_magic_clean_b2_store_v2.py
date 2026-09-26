import hashlib
import io
import threading
import time

from botocore.exceptions import ClientError

from hear.contracts.jobs import ArtifactStorage
from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.storage.magic_clean import B2ImmutableArtifactStore


class Body:
    def __init__(self, data):
        self._stream = io.BytesIO(data)

    def read(self, size=-1):
        return self._stream.read(size)

    def close(self):
        self._stream.close()


class Client:
    def __init__(self):
        self.objects = {}
        self.calls = []

    def put_object(self, **kwargs):
        key = kwargs["Key"]
        data = kwargs["Body"].read()
        self.calls.append(("put", key))
        if key in self.objects:
            raise ClientError(
                {
                    "Error": {"Code": "PreconditionFailed"},
                    "ResponseMetadata": {"HTTPStatusCode": 412},
                },
                "PutObject",
            )
        self.objects[key] = (data, kwargs["ContentType"])
        return {"VersionId": "version-1", "ChecksumSHA256": kwargs["ChecksumSHA256"]}

    def get_object(self, **kwargs):
        data, content_type = self.objects[kwargs["Key"]]
        self.calls.append(("get", kwargs["Key"]))
        return {
            "Body": Body(data),
            "ContentType": content_type,
            "VersionId": "version-1",
        }


def context():
    return ArtifactStorage.model_validate(
        {
            "endpoint_url": "https://s3.example",
            "bucket_name": "bucket",
            "key_id": "key",
            "application_key": "secret",
            "folder_prefix": "tenant/jobs/",
            "public_base_url": "https://cdn.example",
            "expires_at": "2026-09-27T00:00:00Z",
        }
    )


def guard(tmp_path):
    return ResourceGuard(
        ResourceBudget(1024 * 1024, 1024 * 1024, 48000),
        tmp_path,
        time.monotonic() + 60,
        threading.Event(),
    )


def test_store_uses_create_only_upload_without_readback(monkeypatch, tmp_path):
    client = Client()
    monkeypatch.setattr("hear.storage.magic_clean.boto3.client", lambda *args, **kwargs: client)
    store = B2ImmutableArtifactStore(context())
    data = b"artifact"
    digest = hashlib.sha256(data).hexdigest()
    result = store.create(
        "tenant/jobs/a.bin",
        io.BytesIO(data),
        size_bytes=len(data),
        sha256=digest,
        content_type="application/octet-stream",
        guard=guard(tmp_path),
    )
    assert result.version == "version-1"
    assert client.calls == [("put", "tenant/jobs/a.bin")]


def test_store_reconciles_existing_identical_object(monkeypatch, tmp_path):
    client = Client()
    monkeypatch.setattr("hear.storage.magic_clean.boto3.client", lambda *args, **kwargs: client)
    store = B2ImmutableArtifactStore(context())
    data = b"artifact"
    digest = hashlib.sha256(data).hexdigest()
    first = dict(
        key="tenant/jobs/a.bin",
        size_bytes=len(data),
        sha256=digest,
        content_type="application/octet-stream",
        guard=guard(tmp_path),
    )
    store.create(source=io.BytesIO(data), **first)
    store.create(source=io.BytesIO(data), **first)
    assert client.calls == [
        ("put", "tenant/jobs/a.bin"),
        ("put", "tenant/jobs/a.bin"),
        ("get", "tenant/jobs/a.bin"),
    ]
