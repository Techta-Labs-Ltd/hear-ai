"""Canonical execution scope shared with the backend; credentials are hashed, not logged."""

import hashlib
import json
from datetime import datetime


class ExecutionScope:
    @staticmethod
    def digest(value: dict) -> str:
        storage = value["storage"]
        source = value["source"]
        expiry = datetime.fromisoformat(str(storage["expires_at"]).replace("Z", "+00:00"))
        deadline = datetime.fromisoformat(str(value["deadline"]).replace("Z", "+00:00"))
        if expiry.utcoffset() is None or deadline.utcoffset() is None:
            raise ValueError("scope_requires_timezone")
        scope = {
            key: value.get(key)
            for key in (
                "backend_id",
                "job_id",
                "run_id",
                "attempt_id",
                "track_id",
                "user_id",
                "job_type",
                "operation",
            )
        }
        scope.update(
            schema_version=value.get("schema_version", 1),
            options=value.get("options", {}),
            backend_base_url=str(value["backend_base_url"]).rstrip("/"),
            artifact_prefix=value["artifact_prefix"].rstrip("/"),
            source_url=str(source["url"]),
            source_revision=source["revision"],
            source_sha256=source.get("file_sha256"),
            bucket=storage["bucket_name"],
            endpoint=str(storage["endpoint_url"]).rstrip("/"),
            public_base=str(storage["public_base_url"]).rstrip("/"),
            folder=storage["folder_prefix"].rstrip("/"),
            credential_id=storage["key_id"],
            credential_sha256=hashlib.sha256(storage["application_key"].encode()).hexdigest(),
            expiry=expiry.timestamp(),
            deadline=deadline.timestamp(),
        )
        return hashlib.sha256(
            json.dumps(scope, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
