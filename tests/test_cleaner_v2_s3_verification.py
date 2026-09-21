import hashlib
import threading
import time
from datetime import timedelta
from io import BytesIO

import pytest

from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.runtime.cleaner.s3_verification import S3BundleVerifier
from hear.services.magic_clean.artifacts import ArtifactWriter
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode
from tests.test_cleaner_v2_artifacts import MemoryStore, bundle
from tests.test_cleaner_v2_contracts import ticket

__all__ = ["bundle", "ticket"]


class ReadClient:
    def __init__(self, store):
        self.store = store
        self.calls = []
        self.bodies = []
        self.changed = {}
        self.corrupt = False

    def get_object(self, **kwargs):
        self.calls.append(kwargs)
        key = kwargs["Key"]
        data = self.store.objects[key]
        content_type = "application/json"
        if key.endswith(".flac"):
            content_type = "audio/flac"
        elif key.endswith(".mp3"):
            content_type = "audio/mpeg"
        body = BytesIO(data[:-1] + b"!" if self.corrupt else data)
        self.bodies.append(body)
        return {
            "Body": body,
            "ContentLength": len(data),
            "ContentType": content_type,
            "VersionId": "version-1",
            "ResponseMetadata": {"HTTPStatusCode": 200},
            **self.changed,
        }


def test_reads_exact_versions_and_verifies_complete_bundle(bundle):
    store = MemoryStore()
    expected = ArtifactWriter(store).publish(*bundle)
    client = ReadClient(store)
    result = S3BundleVerifier(client, "approved-bucket").verify(
        bundle[0], expected.reference, bundle[3]
    )
    assert result == expected
    assert len(client.calls) == 4
    assert all(call["VersionId"] == "version-1" for call in client.calls)
    assert all(call["Bucket"] == "approved-bucket" for call in client.calls)
    assert all(body.closed for body in client.bodies)


@pytest.mark.parametrize(
    "changed",
    [
        {"VersionId": "wrong"},
        {"ContentLength": 1},
        {"ContentType": "audio/wav"},
        {"ResponseMetadata": {"HTTPStatusCode": 206}},
        {
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "HTTPHeaders": {"x-backblaze-live-read-enabled": "true"},
            }
        },
    ],
)
def test_rejects_wrong_or_unfinished_object_metadata(bundle, changed):
    store = MemoryStore()
    result = ArtifactWriter(store).publish(*bundle)
    client = ReadClient(store)
    client.changed = changed
    with pytest.raises(CleanExecutionError) as error:
        S3BundleVerifier(client, "bucket").verify(bundle[0], result.reference, bundle[3])
    assert error.value.code == ErrorCode.ARTIFACT_CONFLICT
    assert all(body.closed for body in client.bodies)


def test_checks_actual_bytes_not_metadata(bundle):
    store = MemoryStore()
    result = ArtifactWriter(store).publish(*bundle)
    client = ReadClient(store)
    client.corrupt = True
    with pytest.raises(CleanExecutionError) as error:
        S3BundleVerifier(client, "bucket").verify(bundle[0], result.reference, bundle[3])
    assert error.value.code == ErrorCode.ARTIFACT_CONFLICT
    assert len(client.calls) == 1
    assert client.bodies[0].closed


def test_artifact_corruption_prevents_bundle_acceptance(bundle):
    store = MemoryStore()
    result = ArtifactWriter(store).publish(*bundle)
    artifact = result.manifest.artifacts[0]
    store.objects[artifact.object_key] = b"x" * artifact.size_bytes
    assert hashlib.sha256(store.objects[artifact.object_key]).hexdigest() != artifact.sha256
    with pytest.raises(CleanExecutionError):
        S3BundleVerifier(ReadClient(store), "bucket").verify(bundle[0], result.reference, bundle[3])


def test_cancelled_reconciliation_performs_no_network_read(bundle):
    store = MemoryStore()
    result = ArtifactWriter(store).publish(*bundle)
    client = ReadClient(store)
    bundle[3].cancelled.set()
    with pytest.raises(CleanExecutionError) as error:
        S3BundleVerifier(client, "bucket").verify(bundle[0], result.reference, bundle[3])
    assert error.value.code == ErrorCode.CANCELLED
    assert not client.calls


def test_completed_bundle_can_be_reconciled_after_execution_deadline(bundle, monkeypatch):
    store = MemoryStore()
    result = ArtifactWriter(store).publish(*bundle)

    class LaterClock:
        @staticmethod
        def now(zone):
            return bundle[0].deadline + timedelta(hours=1)

    monkeypatch.setattr("hear.runtime.cleaner.resource_guard.datetime", LaterClock)
    # This is a newly authorized reconciliation read budget, not the expired
    # execution guard. The manifest still must have completed before its deadline.
    read_guard = ResourceGuard(
        bundle[3].budget, bundle[3].workspace, time.monotonic() + 10, threading.Event()
    )
    verified = S3BundleVerifier(ReadClient(store), "bucket").verify(
        bundle[0], result.reference, read_guard
    )
    assert verified == result
    assert read_guard.wall_deadline is None
