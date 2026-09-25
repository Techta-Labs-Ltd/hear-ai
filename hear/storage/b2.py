from __future__ import annotations

import hashlib
import json
import mimetypes
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import boto3
from boto3.s3.transfer import TransferConfig

from hear.contracts.jobs import ArtifactStorage
from hear.contracts.outcomes import ArtifactManifest


class B2Storage:
    def __init__(
        self,
        context: ArtifactStorage,
        *,
        multipart_threshold: int = 64 * 1024 * 1024,
        multipart_chunksize: int = 16 * 1024 * 1024,
    ) -> None:
        self._context = context
        self._client = boto3.client(
            "s3",
            endpoint_url=str(context.endpoint_url),
            aws_access_key_id=context.key_id,
            aws_secret_access_key=context.application_key.get_secret_value(),
        )
        self._transfer = TransferConfig(
            multipart_threshold=multipart_threshold,
            multipart_chunksize=multipart_chunksize,
            max_concurrency=4,
            use_threads=True,
        )

    def key(self, *parts: str) -> str:
        clean: list[str] = []
        for raw in parts:
            value = str(raw).strip().strip("/")
            parsed = PurePosixPath(value)
            if not value or parsed.is_absolute():
                raise ValueError("invalid storage key")
            if any(part in {"", ".", ".."} for part in parsed.parts):
                raise ValueError("invalid storage key")
            if "\\" in value or "\x00" in value:
                raise ValueError("invalid storage key")
            clean.extend(parsed.parts)
        return self._context.folder_prefix + "/".join(clean)

    def upload_file(
        self,
        local_path: Path,
        object_key: str,
        *,
        sha256: str,
        content_type: str | None = None,
    ) -> ArtifactManifest:
        self._validate_key(object_key)
        self._validate_digest(sha256)
        resolved_type = (
            content_type
            or mimetypes.guess_type(object_key)[0]
            or mimetypes.guess_type(str(local_path))[0]
            or "application/octet-stream"
        )
        self._client.upload_file(
            str(local_path),
            self._context.bucket_name,
            object_key,
            ExtraArgs={
                "ContentType": resolved_type,
                "Metadata": {"sha256": sha256},
            },
            Config=self._transfer,
        )
        return self._verify(
            object_key,
            size_bytes=local_path.stat().st_size,
            sha256=sha256,
            content_type=resolved_type,
        )

    def upload_json(self, payload: dict, object_key: str) -> ArtifactManifest:
        self._validate_key(object_key)
        data = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(data).hexdigest()
        self._client.put_object(
            Bucket=self._context.bucket_name,
            Key=object_key,
            Body=data,
            ContentType="application/json",
            Metadata={"sha256": digest},
        )
        return self._verify(
            object_key,
            size_bytes=len(data),
            sha256=digest,
            content_type="application/json",
        )

    def _verify(
        self,
        object_key: str,
        *,
        size_bytes: int,
        sha256: str,
        content_type: str,
    ) -> ArtifactManifest:
        head = self._client.head_object(
            Bucket=self._context.bucket_name,
            Key=object_key,
        )
        if int(head.get("ContentLength") or -1) != size_bytes:
            raise RuntimeError("uploaded object size mismatch")
        remote_sha = str((head.get("Metadata") or {}).get("sha256") or "").lower()
        if remote_sha != sha256:
            raise RuntimeError("uploaded object checksum metadata mismatch")
        return ArtifactManifest(
            bucket_name=self._context.bucket_name,
            object_key=object_key,
            size_bytes=size_bytes,
            sha256=sha256,
            content_type=content_type,
            audio_url=self._public_url(object_key),
        )

    def _validate_key(self, object_key: str) -> None:
        if not object_key.startswith(self._context.folder_prefix):
            raise ValueError("object key escapes storage prefix")

    @staticmethod
    def _validate_digest(value: str) -> None:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("invalid sha256")

    def _public_url(self, object_key: str) -> str:
        encoded = "/".join(quote(part, safe="") for part in object_key.split("/"))
        return f"{str(self._context.public_base_url).rstrip('/')}/{encoded}"


class B2StorageFactory:
    def __init__(self, *, multipart_threshold: int = 64 * 1024 * 1024) -> None:
        self._multipart_threshold = multipart_threshold

    def create(self, context: ArtifactStorage) -> B2Storage:
        return B2Storage(
            context,
            multipart_threshold=self._multipart_threshold,
        )