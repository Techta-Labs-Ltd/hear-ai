import hashlib
import json
import threading
import time
from datetime import UTC, datetime, timedelta
from io import BytesIO

import pytest

from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.runtime.cleaner.s3_verification import S3SourceStager
from hear.services.magic_clean.contracts import AttemptTicket, CleanExecutionError, ErrorCode
from tests.test_cleaner_v2_contracts import ticket as ticket_fixture

ticket = ticket_fixture


@pytest.fixture
def staging(tmp_path, ticket):
    data = b"source-fixture" * 100000
    ticket["deadline"] = (datetime.now(UTC) + timedelta(minutes=5)).isoformat()
    ticket["input"].update(size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    parsed = AttemptTicket.model_validate_json(json.dumps(ticket))
    guard = ResourceGuard(
        ResourceBudget(4000000, 2000000, 100000),
        tmp_path,
        time.monotonic() + 30,
        threading.Event(),
    )
    return parsed, data, tmp_path / "source", guard


class SourceClient:
    def __init__(self, ticket, data, destination):
        self.ticket, self.data, self.destination = ticket, data, destination
        self.calls = []
        self.body = None
        self.version = ticket.input.object_version

    def get_object(self, **kwargs):
        assert not self.destination.exists()
        self.calls.append(kwargs)
        self.body = BytesIO(self.data)
        return {
            "Body": self.body,
            "VersionId": self.version,
            "ContentLength": self.ticket.input.size_bytes,
            "ContentType": "application/octet-stream",
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }


def test_stages_exact_pinned_bytes_with_no_partial_destination(staging):
    ticket, data, destination, guard = staging
    client = SourceClient(ticket, data, destination)
    result = S3SourceStager(client, "scoped-bucket").stage(ticket, destination, guard)
    assert result == destination
    assert result.read_bytes() == data
    assert client.calls == [
        {
            "Bucket": "scoped-bucket",
            "Key": ticket.input.object_key,
            "VersionId": ticket.input.object_version,
        }
    ]
    assert client.body.closed
    assert not list(guard.workspace.glob("source-stage-*"))


@pytest.mark.parametrize("failure", ["corrupt", "truncated", "oversized", "version"])
def test_bad_remote_source_leaves_no_staged_file(staging, failure):
    ticket, data, destination, guard = staging
    payload = data
    if failure == "corrupt":
        payload = b"!" + data[1:]
    elif failure == "truncated":
        payload = data[:-1]
    elif failure == "oversized":
        payload = data + b"!"
    client = SourceClient(ticket, payload, destination)
    if failure == "version":
        client.version = "other-version"
    with pytest.raises(CleanExecutionError) as error:
        S3SourceStager(client, "bucket").stage(ticket, destination, guard)
    assert error.value.code == ErrorCode.SOURCE_MISMATCH
    assert not destination.exists()
    assert not list(guard.workspace.glob("source-stage-*"))
    assert client.body.closed


def test_existing_source_is_never_overwritten(staging):
    ticket, data, destination, guard = staging
    destination.write_bytes(b"existing")
    client = SourceClient(ticket, data, destination)
    with pytest.raises(CleanExecutionError) as error:
        S3SourceStager(client, "bucket").stage(ticket, destination, guard)
    assert error.value.code == ErrorCode.ARTIFACT_CONFLICT
    assert destination.read_bytes() == b"existing"
    assert not client.calls


def test_download_budget_rejected_before_network(staging):
    ticket, data, destination, guard = staging
    guard.budget = ResourceBudget(4000000, 100, 100000)
    client = SourceClient(ticket, data, destination)
    with pytest.raises(CleanExecutionError) as error:
        S3SourceStager(client, "bucket").stage(ticket, destination, guard)
    assert error.value.code == ErrorCode.RESOURCE_EXHAUSTED
    assert not client.calls
    assert not destination.exists()
