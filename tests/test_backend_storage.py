from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, datetime, timedelta

import pytest
from cryptography.fernet import Fernet

from hear.config import settings
from hear.core.backend_registry import (
    authenticate_backend,
    backend_registry,
    validate_storage_for_backend,
)
from hear.core.storage import (
    B2Storage,
    StorageCredentialsExpiredError,
    _fernet,
    decrypt_storage_context,
    encrypt_storage_context,
    object_key,
)
from hear.models.schemas import StorageContext
from hear.services.jobs.submission import request_fingerprint


def context(**overrides) -> StorageContext:
    values = {
        "endpoint_url": "https://s3.backend-a.test",
        "bucket_name": "bucket-a",
        "key_id": "key-id-secret",
        "application_key": "application-key-secret",
        "folder_prefix": "users/user-1/jobs/job-1",
        "public_base_url": "https://cdn.backend-a.test/media",
        "expires_at": datetime.now(UTC) + timedelta(hours=2),
    }
    values.update(overrides)
    return StorageContext(**values)


@pytest.fixture
def configured(monkeypatch):
    registry = {
        "backend-a": {
            "service_key_sha256": hashlib.sha256(b"backend-a-secret").hexdigest(),
            "allowed_endpoint_urls": ["https://s3.backend-a.test"],
            "allowed_buckets": ["bucket-a"],
            "allowed_public_base_urls": ["https://cdn.backend-a.test/media"],
        },
        "backend-b": {
            "service_key_sha256": hashlib.sha256(b"backend-b-secret").hexdigest(),
            "allowed_endpoint_urls": ["https://s3.backend-b.test"],
            "allowed_buckets": ["bucket-b"],
            "allowed_public_base_urls": ["https://cdn.backend-b.test/media"],
        },
    }
    monkeypatch.setattr(settings, "BACKEND_REGISTRY_JSON", json.dumps(registry))
    monkeypatch.setattr(settings, "STORAGE_CONTEXT_ENCRYPTION_KEY", Fernet.generate_key().decode())
    backend_registry.cache_clear()
    _fernet.cache_clear()
    yield
    backend_registry.cache_clear()
    _fernet.cache_clear()


def test_service_keys_are_bound_to_exact_backend(configured):
    assert authenticate_backend("backend-a", "backend-a-secret")
    assert not authenticate_backend("backend-b", "backend-a-secret")
    assert not authenticate_backend("backend-a", "wrong")


def test_storage_destination_must_match_backend_allow_list(configured):
    validate_storage_for_backend("backend-a", context())
    with pytest.raises(ValueError, match="endpoint"):
        validate_storage_for_backend(
            "backend-a", context(endpoint_url="https://s3.backend-b.test")
        )
    with pytest.raises(ValueError, match="bucket"):
        validate_storage_for_backend("backend-a", context(bucket_name="bucket-b"))


@pytest.mark.parametrize(
    "prefix",
    ["../escape", "users/./job", "users//job", "/", "users\\escape"],
)
def test_storage_prefix_rejects_traversal(prefix):
    with pytest.raises(ValueError, match="folder_prefix"):
        context(folder_prefix=prefix)


def test_encrypted_context_round_trips_without_plaintext_secrets(configured):
    original = context()
    encrypted = encrypt_storage_context(original)
    assert original.key_id not in encrypted
    assert original.application_key not in encrypted
    assert decrypt_storage_context(encrypted) == original


def test_expired_encrypted_context_has_stable_error(configured):
    expired = context(expires_at=datetime.now(UTC) - timedelta(seconds=1))
    encrypted = encrypt_storage_context(expired)
    with pytest.raises(StorageCredentialsExpiredError) as error:
        decrypt_storage_context(encrypted)
    assert error.value.code == "storage_credentials_expired"
    assert str(error.value) == "storage_credentials_expired"


def test_expired_context_can_be_authenticated_for_credential_refresh(configured):
    expired = context(expires_at=datetime.now(UTC) - timedelta(seconds=1))
    encrypted = encrypt_storage_context(expired)

    assert decrypt_storage_context(encrypted, require_active=False) == expired


def test_object_keys_and_public_urls_stay_under_authorized_prefix(configured):
    storage = object.__new__(B2Storage)
    storage.context = context()
    key = object_key(storage.context, "enhanced", "job-1.mp3")
    assert key == "users/user-1/jobs/job-1/enhanced/job-1.mp3"
    assert storage._public_url(key).startswith("https://cdn.backend-a.test/media/")
    with pytest.raises(ValueError, match="unsafe"):
        object_key(storage.context, "../other-backend", "file.mp3")


def test_upload_verifies_sha256_metadata_and_size(configured, tmp_path):
    payload = b"verified Magic Clean artifact"
    local_path = tmp_path / "artifact.mp3"
    local_path.write_bytes(payload)
    checksum = hashlib.sha256(payload).hexdigest()

    class FakeClient:
        def __init__(self):
            self.extra_args = None
            self.deleted = False

        def upload_file(self, _path, _bucket, _key, *, ExtraArgs):
            self.extra_args = ExtraArgs

        def head_object(self, **_kwargs):
            return {
                "ContentLength": len(payload),
                "Metadata": {"sha256": checksum},
            }

        def get_object(self, **_kwargs):
            return {"Body": io.BytesIO(payload)}

        def delete_object(self, **_kwargs):
            self.deleted = True

    storage = object.__new__(B2Storage)
    storage.context = context()
    storage._client = FakeClient()
    key = storage.key("enhanced", "job-1.mp3")

    url = storage.upload_file(
        str(local_path),
        key,
        "audio/mpeg",
        checksum_sha256=checksum,
    )

    assert storage._client.extra_args["Metadata"] == {"sha256": checksum}
    assert storage._client.deleted is False
    assert url.endswith("/users/user-1/jobs/job-1/enhanced/job-1.mp3")


def test_upload_removes_object_when_checksum_metadata_is_not_verified(
    configured, tmp_path
):
    local_path = tmp_path / "artifact.mp3"
    local_path.write_bytes(b"payload")
    checksum = hashlib.sha256(b"payload").hexdigest()

    class FakeClient:
        deleted = False

        def upload_file(self, *_args, **_kwargs):
            return None

        def head_object(self, **_kwargs):
            return {"ContentLength": 7, "Metadata": {}}

        def get_object(self, **_kwargs):
            return {"Body": io.BytesIO(b"payload")}

        def delete_object(self, **_kwargs):
            self.deleted = True

    storage = object.__new__(B2Storage)
    storage.context = context()
    storage._client = FakeClient()

    with pytest.raises(RuntimeError, match="checksum metadata"):
        storage.upload_file(
            str(local_path),
            storage.key("enhanced", "job-1.mp3"),
            checksum_sha256=checksum,
        )

    assert storage._client.deleted is True


def test_upload_removes_object_when_remote_content_checksum_differs(
    configured,
    tmp_path,
):
    payload = b"verified payload"
    local_path = tmp_path / "artifact.mp3"
    local_path.write_bytes(payload)
    checksum = hashlib.sha256(payload).hexdigest()

    class FakeClient:
        deleted = False

        def upload_file(self, *_args, **_kwargs):
            return None

        def head_object(self, **_kwargs):
            return {
                "ContentLength": len(payload),
                "Metadata": {"sha256": checksum},
            }

        def get_object(self, **_kwargs):
            return {"Body": io.BytesIO(b"corrupted payload")}

        def delete_object(self, **_kwargs):
            self.deleted = True

    storage = object.__new__(B2Storage)
    storage.context = context()
    storage._client = FakeClient()

    with pytest.raises(RuntimeError, match="content checksum"):
        storage.upload_file(
            str(local_path),
            storage.key("enhanced", "job-1.mp3"),
            checksum_sha256=checksum,
        )

    assert storage._client.deleted is True


def test_fingerprint_excludes_credentials_but_includes_destination():
    base = {
        "job_id": "job-1",
        "backend_id": "backend-a",
        "storage": context().model_dump(mode="json"),
    }
    rotated = json.loads(json.dumps(base, default=str))
    rotated["storage"]["key_id"] = "rotated-id"
    rotated["storage"]["application_key"] = "rotated-secret"
    other_prefix = json.loads(json.dumps(base, default=str))
    other_prefix["storage"]["folder_prefix"] = "users/user-1/jobs/job-2/"
    other_backend = json.loads(json.dumps(base, default=str))
    other_backend["backend_id"] = "backend-b"

    assert request_fingerprint(base) == request_fingerprint(rotated)
    assert request_fingerprint(base) != request_fingerprint(other_prefix)
    assert request_fingerprint(base) != request_fingerprint(other_backend)
