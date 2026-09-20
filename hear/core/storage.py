"""Job-scoped, encrypted B2 storage configuration and object operations."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import PurePosixPath
from urllib.parse import quote

import boto3
from botocore.exceptions import ClientError
from cryptography.fernet import Fernet, InvalidToken

from hear.config import settings
from hear.models.schemas import StorageContext


class StorageContextError(RuntimeError):
    code = "invalid_storage_context"


class MissingStorageContextError(StorageContextError):
    code = "missing_storage_context"


class StorageCredentialsExpiredError(StorageContextError):
    code = "storage_credentials_expired"


class StorageCredentialsExpiringError(StorageContextError):
    code = "storage_credentials_expiring"


class StorageContexts:
    @staticmethod
    @lru_cache(maxsize=1)
    def _fernet() -> Fernet:
        value = settings.STORAGE_CONTEXT_ENCRYPTION_KEY.strip().encode("ascii")
        try:
            return Fernet(value)
        except (ValueError, UnicodeError) as exc:
            raise RuntimeError("STORAGE_CONTEXT_ENCRYPTION_KEY must be a valid Fernet key") from exc

    @staticmethod
    def validate_storage_encryption_key(value: str | None = None) -> None:
        if value is None:
            StorageContexts._fernet()
            return
        try:
            Fernet(value.strip().encode("ascii"))
        except (ValueError, UnicodeError) as exc:
            raise RuntimeError("STORAGE_CONTEXT_ENCRYPTION_KEY must be a valid Fernet key") from exc

    @staticmethod
    def encrypt_storage_context(storage: StorageContext) -> str:
        payload = storage.model_dump_json().encode("utf-8")
        return StorageContexts._fernet().encrypt(payload).decode("ascii")

    @staticmethod
    def decrypt_storage_context(
        token: str | None, *, require_active: bool = True
    ) -> StorageContext:
        if not token:
            raise MissingStorageContextError("missing_storage_context")
        try:
            payload = StorageContexts._fernet().decrypt(token.encode("ascii"))
            storage = StorageContext.model_validate(json.loads(payload))
        except MissingStorageContextError:
            raise
        except (InvalidToken, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise StorageContextError("invalid_storage_context") from exc
        expires = storage.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=UTC)
        if require_active and expires <= datetime.now(UTC):
            raise StorageCredentialsExpiredError("storage_credentials_expired")
        return storage

    @staticmethod
    def storage_for_job(job) -> B2Storage:
        return B2Storage(
            StorageContexts.decrypt_storage_context(getattr(job, "storage_context_encrypted", None))
        )

    @staticmethod
    def object_key(storage: StorageContext, *parts: str) -> str:
        clean_parts: list[str] = []
        for raw in parts:
            value = str(raw).strip().strip("/")
            parsed = PurePosixPath(value)
            unsafe_part = any(part in {"", ".", ".."} for part in parsed.parts)
            if not value or parsed.is_absolute() or unsafe_part:
                raise ValueError("unsafe storage object key component")
            if "\\" in value or "\x00" in value:
                raise ValueError("unsafe storage object key component")
            clean_parts.extend(parsed.parts)
        return storage.folder_prefix + "/".join(clean_parts)


class B2Storage:
    def __init__(self, context: StorageContext):
        self.context = context
        self._client = boto3.client(
            "s3",
            endpoint_url=context.endpoint_url,
            aws_access_key_id=context.key_id,
            aws_secret_access_key=context.application_key,
        )

    @property
    def bucket_name(self) -> str:
        return self.context.bucket_name

    def _ensure_active(self) -> None:
        expires = self.context.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=UTC)
        if expires <= datetime.now(UTC):
            raise StorageCredentialsExpiredError("storage_credentials_expired")

    def key(self, *parts: str) -> str:
        return StorageContexts.object_key(self.context, *parts)

    def _public_url(self, remote_key: str) -> str:
        encoded = "/".join(quote(part, safe="") for part in remote_key.split("/"))
        return f"{self.context.public_base_url.rstrip('/')}/{encoded}"

    def artifact(self, remote_key: str, audio_url: str) -> dict[str, str]:
        return {"bucket_name": self.bucket_name, "b2_key": remote_key, "audio_url": audio_url}

    def upload_file(
        self,
        local_path: str,
        remote_key: str,
        content_type: str | None = None,
        *,
        checksum_sha256: str | None = None,
    ) -> str:
        self._ensure_active()
        if not remote_key.startswith(self.context.folder_prefix):
            raise ValueError("object key escapes authorized folder prefix")
        resolved_type = (
            content_type
            or mimetypes.guess_type(remote_key)[0]
            or mimetypes.guess_type(local_path)[0]
            or "application/octet-stream"
        )
        extra_args: dict[str, object] = {"ContentType": resolved_type}
        if checksum_sha256 is not None:
            normalized_checksum = checksum_sha256.strip().lower()
            if len(normalized_checksum) != 64 or any(
                character not in "0123456789abcdef" for character in normalized_checksum
            ):
                raise ValueError("checksum_sha256 must be a lowercase SHA-256 hex digest")
            extra_args["Metadata"] = {"sha256": normalized_checksum}
        try:
            self._client.upload_file(local_path, self.bucket_name, remote_key, ExtraArgs=extra_args)
            uploaded = self._client.head_object(Bucket=self.bucket_name, Key=remote_key)
            local_size = os.path.getsize(local_path)
            remote_size = int(uploaded.get("ContentLength") or -1)
            if remote_size != local_size:
                raise RuntimeError(
                    f"uploaded object size mismatch: local={local_size}, remote={remote_size}"
                )
            if checksum_sha256 is not None:
                remote_checksum = str((uploaded.get("Metadata") or {}).get("sha256") or "").lower()
                if remote_checksum != normalized_checksum:
                    raise RuntimeError("uploaded object checksum metadata mismatch")
                remote_object = self._client.get_object(Bucket=self.bucket_name, Key=remote_key)
                body = remote_object["Body"]
                digest = hashlib.sha256()
                try:
                    while chunk := body.read(1024 * 1024):
                        digest.update(chunk)
                finally:
                    close = getattr(body, "close", None)
                    if callable(close):
                        close()
                if digest.hexdigest() != normalized_checksum:
                    raise RuntimeError("uploaded object content checksum mismatch")
        except Exception:
            try:
                self._client.delete_object(Bucket=self.bucket_name, Key=remote_key)
            except Exception:
                pass
            raise
        return self._public_url(remote_key)

    def delete_object(self, key: str | None) -> None:
        self._ensure_active()
        if not key or not isinstance(key, str) or (not key.strip()):
            return
        clean = key.strip()
        if not clean.startswith(self.context.folder_prefix):
            raise ValueError("object key escapes authorized folder prefix")
        self._client.delete_object(Bucket=self.bucket_name, Key=clean)
        try:
            self._client.head_object(Bucket=self.bucket_name, Key=clean)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code") or "")
            status = int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") or 0)
            if code in {"404", "NoSuchKey", "NotFound"} or status == 404:
                return
            raise
        raise RuntimeError("storage object still exists after deletion")
