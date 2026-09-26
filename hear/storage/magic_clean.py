from __future__ import annotations

import base64
import hashlib

import boto3
from botocore.exceptions import ClientError

from hear.contracts.jobs import ArtifactStorage
from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.services.magic_clean.artifacts import StoredObject
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


class B2ImmutableArtifactStore:
    def __init__(self, context: ArtifactStorage) -> None:
        self._context = context
        self._client = boto3.client(
            "s3",
            endpoint_url=str(context.endpoint_url),
            aws_access_key_id=context.key_id,
            aws_secret_access_key=context.application_key.get_secret_value(),
        )

    @property
    def client(self):
        return self._client

    @property
    def bucket_name(self) -> str:
        return self._context.bucket_name

    def create(
        self,
        key: str,
        source,
        *,
        size_bytes: int,
        sha256: str,
        content_type: str,
        guard: ResourceGuard,
    ) -> StoredObject:
        self._validate_key(key)
        guard.check()
        source.seek(0)
        checksum = base64.b64encode(bytes.fromhex(sha256)).decode()
        try:
            response = self._client.put_object(
                Bucket=self._context.bucket_name,
                Key=key,
                Body=source,
                ContentLength=size_bytes,
                ContentType=content_type,
                Metadata={"sha256": sha256},
                ChecksumSHA256=checksum,
                IfNoneMatch="*",
            )
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code") or "")
            status = int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") or 0)
            if code in {"PreconditionFailed", "412"} or status == 412:
                return self._existing(key, size_bytes, sha256, content_type, guard)
            raise CleanExecutionError(ErrorCode.STORAGE_FAILED, "artifact upload failed") from exc
        guard.check()
        remote_checksum = str(response.get("ChecksumSHA256") or "")
        if remote_checksum and remote_checksum != checksum:
            raise CleanExecutionError(
                ErrorCode.STORAGE_FAILED, "artifact checksum response mismatch"
            )
        version = self._version(response)
        if not version:
            head = self._client.head_object(Bucket=self._context.bucket_name, Key=key)
            version = self._version(head)
        if not version:
            raise CleanExecutionError(ErrorCode.STORAGE_FAILED, "artifact version missing")
        return StoredObject(key, version, sha256, size_bytes)

    def _existing(
        self,
        key: str,
        size_bytes: int,
        sha256: str,
        content_type: str,
        guard: ResourceGuard,
    ) -> StoredObject:
        response = self._client.get_object(Bucket=self._context.bucket_name, Key=key)
        body = response["Body"]
        digest = hashlib.sha256()
        size = 0
        try:
            while True:
                guard.check()
                chunk = body.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > size_bytes:
                    raise CleanExecutionError(
                        ErrorCode.ARTIFACT_CONFLICT, "artifact length conflict"
                    )
                digest.update(chunk)
        finally:
            body.close()
        if (
            size != size_bytes
            or digest.hexdigest() != sha256
            or str(response.get("ContentType") or "") != content_type
        ):
            raise CleanExecutionError(ErrorCode.ARTIFACT_CONFLICT, "artifact content conflict")
        version = self._version(response)
        if not version:
            raise CleanExecutionError(ErrorCode.STORAGE_FAILED, "artifact version missing")
        return StoredObject(key, version, sha256, size_bytes)

    def _validate_key(self, key: str) -> None:
        if not key.startswith(self._context.folder_prefix):
            raise CleanExecutionError(ErrorCode.ARTIFACT_CONFLICT, "artifact key outside grant")

    @staticmethod
    def _version(response: dict) -> str:
        direct = response.get("VersionId")
        if direct:
            return str(direct)
        headers = response.get("ResponseMetadata", {}).get("HTTPHeaders", {})
        return str(headers.get("x-amz-version-id") or "")
